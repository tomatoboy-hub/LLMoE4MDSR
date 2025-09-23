import pickle
import numpy as np
import torch
import torch.nn as nn
from models.BaseModel import BaseSeqModel
from models.SASRec import SASRecBackbone
from models.utils import Contrastive_Loss2, cal_bpr_loss
from itertools import combinations

class LLMoEMDSR_base(BaseSeqModel):
    def __init__(self, user_num, item_num_dict, device,args) -> None:
        self.num_domains = len(item_num_dict)
        self.item_nums = [item_num_dict[i] for i in range(self.num_domains)]
        item_num = sum(self.item_nums)

        super().__init__(user_num, item_num, device, args)
        self.global_emb = args.global_emb

        llm_emb_all = pickle.load(open(f".data/{args.dataset}/handled/{args.llm_emb_file}_all.pkl", "rb"))
        llm_item_emb = np.concatenate([np.zeros((1, llm_emb_all.shape[1])), llm_emb_all])
        
        self.item_emb_llm = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb), padding_idx = 0)
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1]/2)),
            nn.Linear(int(llm_item_emb.shape[1]/2), args.hidden_size)
        )

        self.pos_emb = nn.Embedding(args.max_len + 1, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.backbone = SASRecBackbone(device, args)

        self.local_item_embs = nn.ModuleList()
        self.local_pos_embs = nn.ModuleList()
        self.local_emb_dropouts = nn.ModuleList()
        self.local_backbones = nn.ModuleList()

        for i in range(self.num_domains):
            domain_char = str(i)
            llm_emb_local = pickle.load(open(f".data/{args.dataset}/handled/{args.llm_emb_file}_{domain_char}.pkl", "rb"))
            llm_emb_local = np.concatenate([np.zeros((1,llm_emb_local.shape[1])),llm_emb_local])

            if args.local_emb:
                self.local_item_embs.append(nn.Embedding.from_pretrained(torch.Tensor(llm_emb_local), padding_idx = 0))
            else:
                self.local_item_embs.append(nn.Embedding(self.item_nums[i] + 1, args.hidden_size, padding_idx = 0))
            
            self.local_pos_embs.append(nn.Embedding(args.max_len + 1, args.hidden_size))
            self.local_emb_dropouts.append(nn.Dropout(p=args.dropout_rate))
            self.local_backbones.append(SASRecBackbone(device,args))
        
        self.loss_func = nn.BCEWithLogitsLoss()
    
    def _get_embedding(self, item_ids, domain_id):
        if domain_id == "global": 
            if self.global_emb:
                item_seq_emb = self.item_emb_llm(item_ids)
                item_seq_emb = self.adapter(item_seq_emb)
            else:
                item_seq_emb = self.item_emb_llm(item_ids)
        else: 
            item_seq_emb = self.local_item_embs[domain_id](item_ids)
        return item_seq_emb

    def log2feats(self,log_seqs,positions,domain_id):
        if domain_id == "global":
            seqs = self._get_embedding(log_seqs,domain_id="global")
            seqs *= self.item_emb_llm.embedding_dim ** 0.5
            seqs += self.pos_emb(positions.long())
            seqs = self.emb_dropout(seqs)
            log_feats = self.backbone(seqs,log_seqs)
        else:
            seqs = self._get_embedding(log_seqs, domain_id=domain_id)
            seqs *= self.local_item_embs[domain_id].embedding_dim ** 0.5
            seqs += self.local_pos_embs[domain_id](positions.long())
            seqs = self.local_emb_dropouts[domain_id](seqs)
            log_feats = self.local_backbones[domain_id](seqs,log_seqs)
        return log_feats
        
    def forward(self,seq,pos,neg,positions,
                local_seqs, local_poses, local_negs, local_positions,
                target_domain,domain_mask, **kwargs):
            
        log_feats_global = self.log2feats(seq,positions,domain_id="global")
        pos_embs_global = self._get_embedding(pos,domain_id="global")
        neg_embs_global = self._get_embedding(neg,domain_id="global")

        pos_logits_global = (log_feats_global * pos_embs_global).sum(dim=-1)
        neg_logits_global = (log_feats_global * neg_embs_global).sum(dim=-1)

        domain_losses = []
        for i in range(self.num_domains):
            seq_d,pos_d, neg_d,pos_d_positions = local_seqs[i], local_poses[i], local_negs[i], local_positions[i]
            pos_embs_d = self._get_embedding(pos_d,domain_id=i)
            neg_embs_d = self._get_embedding(neg_d,domain_id=i)

            log_feats_d = self.log2feats(seq_d,pos_d_positions,domain_id=i)

            pos_logits_d = (log_feats_d * pos_embs_d).sum(dim=-1)
            neg_logits_d = (log_feats_d * neg_embs_d).sum(dim=-1)

            domain_indices = (pos_d > 0) & (domain_mask == i)
            pos_logits_d[pos_d > 0][domain_mask[pos_d > 0] == i] += pos_logits_global[domain_indices]
            neg_logits_d[pos_d > 0][domain_mask[pos_d > 0] == i] += neg_logits_global[domain_indices]
            
            pos_labels_d = torch.ones_like(pos_logits_d,device=self.device)
            neg_labels_d = torch.zeros_like(neg_logits_d,device=self.device)
            indices_d = (pos_d != 0)

            pos_loss_d = self.loss_func(pos_logits_d[indices_d],pos_labels_d[indices_d])
            neg_loss_d = self.loss_func(neg_logits_d[indices_d],neg_labels_d[indices_d])

            loss_d = pos_loss_d + neg_loss_d
            domain_losses.append(loss_d.mean())

        loss = sum(domain_losses)
        return loss

    def predict(self,seq,item_indices,positions,
                local_seqs, local_item_indices, local_positions,
                target_domain, **kwargs):
        log_feats_global = self.log2feats(seq,positions,domain_id = "global")
        final_feat_global = log_feats_global[:,-1,:]
        item_embs_global = self._get_embedding(item_indices, domain_id = "global")
        logits_global = item_embs_global.matmul(final_feat_global.unsqueeze(-1)).squeeze(-1)
        for i in range(self.num_domains):
            seq_d, items_d, pos_d = local_seqs[i], local_item_indices[i],local_positions[i]

            log_feats_d = self.log2feats(seq_d,pos_d,domain_id = i)
            final_feat_d = log_feats_d[:,-1,:]
            item_embs_d = self._get_embedding(items_d, domain_id = i)
            logits_d = item_embs_d.matmul(final_feat_d.unsqueeze(-1)).squeeze(-1)

            logits_global[target_domain == i] += logits_d[target_domain == i]
        return logits_global
        

class LLMoEMDSR(LLMoEMDSR_base):
    def __init__(self, user_num, item_num_dict, device, args):
        super().__init__(user_num, item_num_dict, device, args)

        self.alpha = args.alpha
        self.beta = args.beta

        llm_user_emb = pickle.load(open(f".data/{args.dataset}/handled/{args.user_emb_file}.pkl", "rb"))
        self.user_emb_llm = nn.Embedding.from_pretrained(torch.Tensor(llm_user_emb), padding_idx = 0)
        self.user_emb_llm.weight.requires_grad = False

        self.user_adapter = nn.Sequential(
            nn.Linear(llm_user_emb.shape[1], int(llm_user_emb.shape[1]/2)),
            nn.Linear(int(llm_user_emb.shape[1]/2),args.hidden_size)
        )
        self.reg_loss_func = Contrastive_Loss2(tau = args.tau_reg)
        self.user_loss_func = Contrastive_Loss2(tau = args.tau)
        if "user_emb_file" not in self.filter_init_modules:
            self.filter_init_modules.append(self.user_emb_llm)
        self._init_weights()
    
    def forward(self,reg_list, user_id, **kwargs):
        loss = super().forward(**kwargs)

        total_reg_loss = 0.0
        domain_pairs = combinations(range(self.num_domains),2)
        num_pairs = 0

        for i, j in domain_pairs:
            reg_i_ids = reg_list[i]
            reg_j_ids = reg_list[j]

            reg_i_ids = reg_i_ids[reg_i_ids > 0]
            reg_j_ids = reg_j_ids[reg_j_ids > 0]
            
            if len(reg_i_ids) == 0 or len(reg_j_ids) == 0:
                continue

            reg_i_emb = self._get_embedding(reg_i_ids, domain_id = "global")
            reg_j_emb = self._get_embedding(reg_j_ids, domain_id = "global")

            total_reg_loss += self.reg_loss_func(reg_i_emb,reg_j_emb)
            num_pairs += 1
        
        if num_pairs > 0:
            total_reg_loss /= num_pairs
        
        loss += self.alpha * total_reg_loss

        seq = kwargs['seq']
        positions = kwargs["positions"]

        log_feats = self.log2feats(seq,positions,domain_id = "global")
        final_feat = log_feats[:,-1,:]
        llm_feats = self.user_emb_llm(user_id)
        llm_feats = self.user_adapter(llm_feats)
        user_loss = self.user_loss_func(llm_feats,final_feat)
        loss += self.beta * user_loss

        return loss