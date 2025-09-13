# here put the import lib
import os
import pickle
import torch
from tqdm import tqdm
from trainers.sequence_trainer import SeqTrainer
from models.LLMoEMDSR import LLMoEMDSR
from utils.utils import record_csv, metric_report, metric_domain_report



class MDRTrainer(SeqTrainer):

    def __init__(self, args, logger, writer, device, generator):
        super().__init__(args, logger, writer, device, generator)
        self.num_domains = self.generator.get_num_domains()

    def _create_model(self):

        self.item_num_dict = self.generator.get_item_num_dict()

        if self.args.model_name == "llmoemdsr":
            self.model = LLMoEMDSR(self.user_num, self.item_num_dict, self.device, self.args)
        else:
            raise ValueError
        
        self.model.to(self.device)

    

    def eval(self, epoch=0, test=False):

        print('')
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("********** Running test **********")
            desc = 'Testing'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
            self.model.load_state_dict(model_state_dict['state_dict'])
            self.model.to(self.device)
            test_loader = self.test_loader
        
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info("********** Epoch: %d eval **********" % epoch)
            desc = 'Evaluating'
            test_loader = self.valid_loader
        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)
        target_domain = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):
            inputs = self._prepare_eval_inputs(batch)
            target_domain = torch.cat([target_domain, inputs["target_domain"]])
            
            with torch.no_grad():
                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]],dim=1)
                local_item_indices = []
                for i in range(self.num_domains):
                    pos_d = inputs["local_poses"][i]
                    neg_d = inputs["local_negs"][i]
                    local_item_indices.append(torch.cat[pos_d.unsqueeze(1),neg_d],dim=1)
                inputs["local_item_indices"] = local_item_indices
                pred_logits = -self.model.predict(**inputs)
                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:,0]
                pred_rank = torch.cat([pred_rank,per_pred_rank])

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        domain_res_dicts = {}
        for i in range(self.num_domains):
            domain_char = i
            pred_rank_d = pred_rank[target_domain == domain_char]
            if len(pred_rank_d) > 0:
                res_dict_d = metric_domain_report(pred_rank_d.detach().cpu().numpy(),domain=str(i))
                domain_res_dicts[domain_char] = res_dict_d
        
        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('\t %s: %.5f' % (k, v))
        
        if test:
            for domain_char, res_d in domain_res_dicts.items():
                self.logger.info(f"Domain {domain_char} Performance:")
                for k, v in res_d.items():
                    self.logger.info(f'\t {k}: {v:.5f}')
        for res_d in domain_res_dicts.values():
            res_dict.update(res_d)
        
        if test:
            record_csv(self.args, res_dict)
        return res_dict

    def save_item_emb(self):

        model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
        try:
            self.model.load_state_dict(model_state_dict['state_dict'])
        except:
            self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)

        all_index = torch.arange(start=1, end=self.item_num+1).to(self.device)
        item_emb = self.model._get_embedding(all_index, domain_id = "global")
        item_emb = item_emb.detach().cpu().numpy()

        output_path = f"./data/{self.args.dataset}/handled/itm_emb_{self.args.model_name}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(item_emb, f)
        self.logger.info(f"Item embeddings saved to {output_path}")


    def eval_cold(self):

        print('')
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running cold test **********")
        desc = 'Testing'
        model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
        self.model.load_state_dict(model_state_dict['state_dict'])
        self.model.to(self.device)
        test_loader = self.generator.make_coldloader()
    
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        target_domain = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):
            inputs = self._prepare_eval_inputs(batch)
            target_domain = torch.cat([target_domain, inputs["target_domain"]])
            with torch.no_grad():
                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]],dim=1)
                local_item_indices = []

                for i in range(self.num_domains):
                    pos_d = inputs["local_poses"][i]
                    neg_d = inputs["local_negs"][i]
                    local_item_indices.append(torch.cat([pos_d.unsqueeze(1),neg_d],dim=1))
        
                inputs["local_item_indices"] = local_item_indices
                pred_logits = -self.model.predict(**inputs)
                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())

        domain_res_dicts = {}
        for i in range(self.num_domains):
            domain_char = i
            pred_rank_d = pred_rank[target_domain == domain_char]
            if len(pred_rank_d) > 0:
                res_dict_d = metric_domain_report(pred_rank_d.detach().cpu().numpy(),domain=domain_char)
                domain_res_dicts[domain_char] = res_dict_d


        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            self.logger.info('\t %s: %.5f' % (k, v))
        
        for res_d in domain_res_dicts.values():
            res_dict.update(res_d)

        key_list = list(res_dict.keys())
        for key in key_list:
            res_dict.update({"cold_{}".format(key): res_dict.pop(key)})

        record_csv(self.args, res_dict)
        
        return res_dict

