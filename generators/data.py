# here put the import lib
import copy
import numpy as np
from torch.utils.data import Dataset
from utils.utils import random_neq
from generators.data_utils import truncate_padding
from itertools import combinations



class SeqDataset(Dataset):
    '''The train dataset for Sequential recommendation'''
    def __init__(self, data, item_num, max_len, neg_num=1):
        
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.var_name = ["seq", "pos", "neg", "positions"]


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        non_neg = list(non_neg)
        pos = inter[-1]
        neg = []
        for _ in range(self.neg_num):
            per_neg = random_neq(1, self.item_num+1, non_neg)
            neg.append(per_neg)
            non_neg.append(per_neg)
        neg = np.array(neg)
        #neg = random_neq(1, self.item_num+1, inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, pos, neg, positions



class CDSRSeq2SeqDataset(Dataset):
    '''The train dataset for Sequential recommendation with seq-to-seq loss'''

    def __init__(self, args, data, domain_data, item_num_dict, max_len, neg_num=1):
        
        super().__init__()
        self.data = data
        self.domain_data = domain_data
        self.item_numA = item_num_dict["0"]
        self.item_numB = item_num_dict["1"]
        self.item_num = self.item_numA + self.item_numB
        self.max_len = max_len
        self.neg_num = neg_num
        self.aug_seq = args.aug_seq
        self.aug_seq_len = args.aug_seq_len
        self.var_name = ["seq", "pos", "neg", "positions",
                         "seqA", "posA", "negA", "positionsA",
                         "seqB", "posB", "negB", "positionsB",
                         "target_domain", "domain_mask"]


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        inter = self.data[index]
        inter = np.array(inter)
        domain_mask = self.domain_data[index]

        all_inter = copy.deepcopy(inter)
        all_inter[np.where(domain_mask==1)] = all_inter[np.where(domain_mask==1)] + self.item_numA
        target_domain = domain_mask[-1]

        seq, pos, neg, positions, mask = truncate_padding(all_inter, domain_mask, self.max_len, self.item_numA, self.item_numB)

        # get the sequence of each domain by mask
        interA = inter[-self.max_len-1:][np.where(domain_mask[-self.max_len-1:]==0)]
        domain_maskA = domain_mask[-self.max_len-1:][np.where(domain_mask[-self.max_len-1:]==0)]
        seqA, posA, negA, positionsA, _ = truncate_padding(interA, domain_maskA, self.max_len, self.item_numA, self.item_numB)
        interB = inter[-self.max_len-1:][np.where(domain_mask[-self.max_len-1:]==1)]
        domain_maskB = np.zeros_like(domain_mask[-self.max_len-1:][np.where(domain_mask[-self.max_len-1:]==1)])   # get neg from index 0 for domain B only
        seqB, posB, negB, positionsB, _ = truncate_padding(interB, domain_maskB, self.max_len, self.item_numB, self.item_numB)  # item_numB replaces item_numA

        # first is domain A, then domain B will lack one positive
        if domain_mask[-self.max_len-1:][0] == 0 and (mask==1).sum() != 0: 
            index = np.where(mask==1)[0][0] # take out the first
            mask[index] = -1
        elif domain_mask[-self.max_len-1:][0] == 1 and (mask==0).sum() != 0: 
            index = np.where(mask==0)[0][0] # take out the first
            mask[index] = -1

        return seq, pos, neg, positions, \
               seqA, posA, negA, positionsA, \
               seqB, posB, negB, positionsB, \
               target_domain, mask


class CDSREvalSeq2SeqDataset(Dataset):
    '''The train dataset for Sequential recommendation with seq-to-seq loss'''

    def __init__(self, args, data, domain_data, item_num_dict, max_len, neg_num=1):
        
        super().__init__()
        self.data = data
        self.domain_data = domain_data
        self.item_numA = item_num_dict["0"]
        self.item_numB = item_num_dict["1"]
        self.item_num = self.item_numA + self.item_numB
        self.max_len = max_len
        self.neg_num = neg_num
        self.aug_seq = args.aug_seq
        self.aug_seq_len = args.aug_seq_len
        self.var_name = ["seq", "pos", "neg", "positions",
                         "seqA", "posA", "negA", "positionsA",
                         "seqB", "posB", "negB", "positionsB",
                         "target_domain", "domain_mask"]


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        inter = self.data[index]
        domain_mask = self.domain_data[index]
        domain_mask = np.array(domain_mask)
        target_domain = domain_mask[-1]

        inter = np.array(inter)
        all_inter = copy.deepcopy(inter)
        all_inter[np.where(domain_mask==1)] = all_inter[np.where(domain_mask==1)] + self.item_numA
        all_inter = list(all_inter)

        pos = all_inter[-1]
        neg, negA, negB = [], [], []
        non_neg = copy.deepcopy(all_inter)
        non_neg_A = list(inter[np.where(domain_mask==0)])
        non_neg_B = list(inter[np.where(domain_mask==1)])
        for _ in range(self.neg_num):
            if target_domain == 0:
                per_neg = random_neq(1, self.item_numA, non_neg)
            elif target_domain == 1:
                per_neg = random_neq(self.item_numA+1, self.item_numA+self.item_numB, non_neg)
            else:
                raise ValueError
            neg.append(per_neg)
            non_neg.append(per_neg)

            per_neg_A = random_neq(1, self.item_numA, non_neg_A)
            negA.append(per_neg_A)
            non_neg_A.append(per_neg_A)

            per_neg_B = random_neq(1, self.item_numB, non_neg_B)
            negB.append(per_neg_B)
            non_neg_B.append(per_neg_B)

        neg, negA, negB = np.array(neg), np.array(negA), np.array(negB)

        seq, _, _, positions, mask = truncate_padding(all_inter, domain_mask, self.max_len, self.item_numA, self.item_numB)

        # get positive for domain A and B
        if len(np.where(domain_mask==0)[0]) == 0:   # no domain A data
            posA = 0
        else:
            posA = inter[np.where(domain_mask==0)][-1]

        if len(np.where(domain_mask==1)[0]) == 0:
            posB = 0
        else:
            posB = inter[np.where(domain_mask==1)][-1]

        # get the sequence of each domain by mask
        inter = np.array(inter)
        interA = inter[-self.max_len-1:][np.where(domain_mask[-self.max_len-1:]==0)]
        domain_maskA = domain_mask[-self.max_len-1:][np.where(domain_mask[-self.max_len-1:]==0)]
        seqA, _, _, positionsA, _ = truncate_padding(interA, domain_maskA, self.max_len, self.item_numA, self.item_numB)
        interB = inter[-self.max_len-1:][np.where(domain_mask[-self.max_len-1:]==1)]
        domain_maskB = np.zeros_like(domain_mask[-self.max_len-1:][np.where(domain_mask[-self.max_len-1:]==1)])   # get neg from index 0 for domain B only
        seqB, _, _, positionsB, _ = truncate_padding(interB, domain_maskB, self.max_len, self.item_numB, self.item_numB)

        # first is domain A, then domain B will lack one positive
        if domain_mask[-self.max_len-1:][0] == 0 and (mask==1).sum() != 0: 
            index = np.where(mask==1)[0][0] # take out the first
            mask[index] = -1
        elif domain_mask[-self.max_len-1:][0] == 1 and (mask==0).sum() != 0: 
            index = np.where(mask==0)[0][0] # take out the first
            mask[index] = -1
        
        return seq, pos, neg, positions, \
               seqA, posA, negA, positionsA, \
               seqB, posB, negB, positionsB, \
               target_domain, mask
        


class CDSRRegSeq2SeqDatasetUser(CDSRSeq2SeqDataset):

    def __init__(self, args, data, domain_data, item_num_dict, max_len, neg_num=1):
        
        super().__init__(args, data, domain_data, item_num_dict, max_len, neg_num)
        
        self.var_name = ["seq", "pos", "neg", "positions",
                         "seqA", "posA", "negA", "positionsA",
                         "seqB", "posB", "negB", "positionsB",
                         "target_domain", "domain_mask",
                         "reg_A", "reg_B", "user_id"]

    def __getitem__(self, index):

        seq, pos, neg, positions, \
        seqA, posA, negA, positionsA, \
        seqB, posB, negB, positionsB, \
        target_domain, mask = super().__getitem__(index)

        inter = self.data[index][:-1]
        domain_mask = copy.deepcopy(self.domain_data[index][:-1])

        inter = np.array(inter)
        domain_mask = np.array(domain_mask)
        inter[np.where(domain_mask==1)] = inter[np.where(domain_mask==1)] + self.item_numA

        index_A = np.where(domain_mask==0)[0]
        index_B = np.where(domain_mask==1)[0]

        if len(index_A) == 0 or len(index_B) == 0:
            reg_A, reg_B = np.array([0]), np.array([0])
        else:
            reg_index_A = np.random.choice(index_A, 1)
            reg_index_B = np.random.choice(index_B, 1)
            reg_A = inter[reg_index_A]
            reg_B = inter[reg_index_B]

        return seq, pos, neg, positions, \
               seqA, posA, negA, positionsA, \
               seqB, posB, negB, positionsB, \
               target_domain, mask, \
               reg_A, reg_B, index

class MDRSeq2SeqDataset(Dataset):
    def __init__(self,args,data,domain_data,item_num_dict,max_len,neg_num=1):
        super().__init__()
        self.data = data
        self.domain_data = domain_data
        self.max_len = max_len
        self.neg_num = neg_num

        self.num_domains = len(item_num_dict)
        self.item_nums = [item_num_dict[str(i)] for i in range(self.num_domains)]
        self.domain_offsets = np.array([sum(self.item_nums[:i]) for i in range(self.num_domains)])
        self.total_item_num = sum(self.item_nums)

        self.var_name = ["seq", "pos", "neg", "positions",
                         "local_seqs", "local_poses", "local_negs", "local_positions",
                         "target_domain", "domain_mask"]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        inter = np.array(self.data[index])
        domain_mask = np.array(self.domain_data[index])
        all_inter = inter + self.domain_offsets[domain_mask]
        target_domain = domain_mask[-1]

        seq, pos, neg, positions, mask = truncate_padding(all_inter, domain_mask, self.max_len, self.total_item_num)
        local_seqs, local_poses, local_negs, local_positions = [], [], [], []

        for i in range(self.num_domains):
            inter_d = inter[domain_mask == i]
            dummy_domain_mask = np.zeros_like(inter_d, dtype = int)
            seq_d,pos_d, neg_d, pos_d_positions,_ = truncate_padding(inter_d,dummy_domain_mask, self.max_len, self.item_nums[i])
            local_seqs.append(seq_d)
            local_poses.append(pos_d)
            local_negs.append(neg_d)
            local_positions.append(pos_d_positions)
        
        return (seq,pos,neg,positions,local_seqs,local_poses,local_negs,local_positions,target_domain,mask)


class MDRRegSeq2SeqDatasetUser(MDRSeq2SeqDataset):
    def __init__(self,args,data, domain_data, item_num_dict, max_len, num_neg=1):
        super().__init__(args,data, domain_data, item_num_dict, max_len, num_neg)

        self.var_name.extend(["reg_list","user_id"])

    def __getitem__(self,index):
        
        base_data_tuple = super().__getitem__(index)
        inter = np.array(self.data[index][:-1])
        domain_mask = np.array(self.domain_data[index][:-1])
        reg_list = []

        for i in range(self.num_domains):
            items_in_domain = inter[domain_mask == i]
            if len(items_in_domain) == 0:
                reg_list.append(np.array([0]))
            else:
                random_item = np.random.choice(items_in_domain,1)
                reg_list.append(random_item + self.domain_offsets[i])
        return (*base_data_tuple,reg_list,index)
    
class MDREvalSeq2SeqDataset(MDRSeq2SeqDataset):
    def __init__(self,args,data,domain_data,item_num_dict, max_len, neg_num = 1):
        super().__init__(args,data,domain_data, item_num_dict, max_len, neg_num)
        self.neg_num = neg_num
    def __getitem__(self,index):
        inter = np.array(self.data[index])
        domain_mask = np.array(self.domain_data[index])
        target_domain = int(domain_mask[-1])

        all_inter = list(inter + self.domain_offsets[domain_mask])
        pos = all_inter[-1]

        non_neg = set(all_inter)
        target_domain_start_id = self.domain_offsets[target_domain] + 1
        target_domain_end_id = self.domain_offsets[target_domain] + self.item_nums[target_domain]

        neg = []
        for _ in range(self.neg_num):
            n = np.random.randint(target_domain_start_id,target_domain_end_id + 1)
            while n in non_neg:
                n = np.random.randint(target_domain_start_id,target_domain_end_id + 1)
            neg.append(n)
        neg = np.array(neg)

        local_seqs, local_poses, local_negs, local_positions = [], [], [], []

        for i in range(self.num_domains):
            inter_d = inter[domain_mask == i]
            dummy_domain_mask = np.zeros_like(inter_d,dtype=int)
            non_neg_d = set(inter_d)

            pos_d = inter_d[-1] if len(inter_d) > 0 else 0
            neg_d = []
            for _ in range(self.neg_num):
                n = np.random.randint(1, self.item_nums[i] + 1)
                while n in non_neg_d:
                    n = np.random.randint(1,self.item_nums[i] + 1)
                neg_d.append(n)
            seq_d,_,_,pos_d_positions,_ = truncate_padding(inter_d,dummy_domain_mask, self.max_len, self.item_nums[i])
            local_seqs.append(seq_d)
            local_poses.append(pos_d)
            local_negs.append(neg_d)
            local_positions.append(pos_d_positions)

        seq,_,_,positions,mask = truncate_padding(all_inter, domain_mask, self.max_len, self.total_item_num)
        return (seq,pos,neg,positions,local_seqs,local_poses, local_negs, local_positions,target_domain,mask)
    
        
        