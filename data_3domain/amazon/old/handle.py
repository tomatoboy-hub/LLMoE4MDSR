#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import json
from data_process import New_Amazon, Amazon_meta
from collections import defaultdict


# 把两个独立的Amazon数据读进来

# In[ ]:


domain_A = "Clothing_Shoes_and_Jewelry"
domain_B = "Sports_and_Outdoors"
domain_C = "AMAZON_FASHION"


# In[ ]:


domain_A = "Clothing_Shoes_and_Jewelry"
domain_B = "Sports_and_Outdoors"
domain_C = "AMAZON_FASHION"
data_A = New_Amazon(domain_A, 0)
data_B = New_Amazon(domain_B, 0)
data_C = New_Amazon(domain_C, 0)


# In[ ]:


# 给每个交互添加domain id
new_data_A, new_data_B, new_data_C = [], [], []
for inter in tqdm(data_A):
    new_inter = list(inter)
    new_inter.append(0)
    new_data_A.append(new_inter)
for inter in tqdm(data_B):
    new_inter = list(inter)
    new_inter.append(1)
    new_data_B.append(new_inter)
for inter in tqdm(data_C):
    new_inter = list(inter)
    new_inter.append(2)
    new_data_C.append(new_inter)


# 1. read_data: 把所有数据读出来，然后存到一个list中
# 2. filter: 单纯过滤掉交互过小的交互，返回的还是list
# 3. id_map: 制作user和item的dict映射，并拆掉list，变成一个用户的交互序列

# In[4]:


def count_inter(data, t_min, t_max):
    
    user_count = {}
    item_count = {}
    for inter in data:
        user_id, item_id, time, _ = inter
        
        if user_id not in user_count.keys():
            user_count[user_id] = 1
        else:
            if time > t_min and time < t_max:
                user_count[user_id] += 1

        if item_id not in item_count.keys():
            item_count[item_id] = 1
        else:
            if time > t_min and time < t_max:
                item_count[item_id] += 1

    
    return user_count, item_count


# In[ ]:


def filter(data, user_minmum, item_minimum, t_min=1451577600, t_max=1459440000):   # 过滤掉交互少的数据
    
    user_count, item_count = count_inter(data, t_min=t_min, t_max=t_max)
    domain_set = {0: {"user": [], "item": []},
                  1: {"user": [], "item": []},
                  2: {"user": [], "item": []},
                  }
    new_data = []

    for inter in tqdm(data):
        user_id, item_id, time, domain_id = inter
        
        if item_count[item_id] > item_minimum and user_count[user_id] > user_minmum \
           and time > t_min and time < t_max:    # 只取2016-01-01到2016-01-15之间的数据
            
            new_data.append(inter)
            domain_set[domain_id]["user"].append(user_id)
            domain_set[domain_id]["item"].append(item_id)
    
    print("filter done!")

    return new_data, domain_set


# In[6]:


def make_sequence(data):

    seq = {}
    domain_seq = {}

    for inter in tqdm(data):
        user_id, item_id, time, domain_id = inter
        if user_id not in seq.keys():
            seq[user_id] = [item_id]
            domain_seq[user_id] = [domain_id]
        else:
            seq[user_id].append(item_id)
            domain_seq[user_id].append(domain_id)

    return seq, domain_seq


# In[ ]:


def id_map(data, domain_set):
    
    final_data, final_domain = {}, {}
    temp_data = {}
    new_user_id = 1
    temp_item_count = {domain_id: len(set(domain_set[domain_id]["item"])) for domain_id in domain_set.keys()}
    item_count = {0: 1, 1: 1, 2: 1}
    item_dict = {
        0: {"str2id": {}, "id2str": {},},
        1: {"str2id": {}, "id2str": {},},
        2: {"str2id": {}, "id2str": {},},
    }
    user_dict = {"str2id": {}, "id2str": {},}

    for inter in tqdm(data):
        user_id, item_id, time, domain_id = inter
            
        if item_id not in item_dict[domain_id]["str2id"].keys():
            new_item_id = item_count[domain_id]
            item_dict[domain_id]["str2id"][item_id] = new_item_id
            item_dict[domain_id]["id2str"][new_item_id] = item_id
            item_count[domain_id] += 1
        
        if user_id not in user_dict["str2id"].keys():
            user_dict["str2id"][user_id] = new_user_id
            user_dict["id2str"][new_user_id] = user_id
            temp_data[new_user_id] = [(item_dict[domain_id]["str2id"][item_id], domain_id, time)]
            new_user_id += 1
        else:
            temp_data[user_dict["str2id"][user_id]].append((item_dict[domain_id]["str2id"][item_id], domain_id, time))

    print("map done!")

    for user_id, inter in tqdm(temp_data.items()):

        inter.sort(key=lambda x: x[2])
        final_data[user_id] = [temp_tuple[0] for temp_tuple in inter]
        final_domain[user_id] = [temp_tuple[1] for temp_tuple in inter]

    print("sort done!")
    
    return final_data, final_domain, user_dict, item_dict, item_count


# In[18]:


# K-core user_core item_core
def check_Kcore(data, user_core, item_core):

    user_count = {}
    item_count = {}
    for inter in data:
        user_id, item_id, time, _ = inter
        
        if user_id not in user_count.keys():
            user_count[user_id] = 1
        else:
            user_count[user_id] += 1

        if item_id not in item_count.keys():
            item_count[item_id] = 1
        else:
            item_count[item_id] += 1

    for _, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for _, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
        
    return user_count, item_count, True


# In[ ]:


# 循环过滤 K-core
def filter_Kcore(data, user_core, item_core): # user 接所有items
    
    user_count, item_count, isKcore = check_Kcore(data, user_core, item_core)
    
    new_data = data

    while not isKcore:

        temp_data = []
        domain_set = {
            0: {"user": [], "item": []},
            1: {"user": [], "item": []},
            2: {"user": [], "item": []},
        }

        for inter in tqdm(new_data):
            user_id, item_id, time, domain_id = inter
            
            if item_count[item_id] > item_core and user_count[user_id] > user_core:    # 只取2016-01-01到2016-01-15之间的数据
                
                temp_data.append(inter)
                domain_set[domain_id]["user"].append(user_id)
                domain_set[domain_id]["item"].append(item_id)
        user_count, item_count, isKcore = check_Kcore(temp_data, user_core, item_core)

        new_data = temp_data

    print("K-core filter done!")

    return new_data, domain_set


# In[34]:


def filter_time(data, t_min=1451577600, t_max=1459440000):   # 过滤掉交互少的数据
    

    new_data = []

    for inter in tqdm(data):
        _, _, time, _ = inter
        
        if time > t_min and time < t_max:    # 只取2016-01-01到2016-01-15之间的数据
            
            new_data.append(inter)

    print("filter time done!")

    return new_data


# In[ ]:


data_A[0]


# In[31]:


time_list = []
for inter in tqdm(new_data_A+new_data_B):
    _, _, time, _ = inter
    time_list.append(time)


# In[ ]:


plt.hist(time_list, bins=30)


# In[ ]:


all_data = new_data_A + new_data_B + new_data_C
# new_data, domain_set = filter(all_data, user_minmum=10, item_minimum=10)
all_data = filter_time(all_data, t_min=1514736000, t_max=1577808000)
new_data, domain_set = filter_Kcore(all_data, user_core=5, item_core=3)
final_data, final_domain, user_dict, item_dict, item_count = id_map(new_data, domain_set)
item_count = {domain_id: len(set(domain_set[domain_id]["item"])) for domain_id in domain_set.keys()}
item_dict["item_count"] = item_count


# In[ ]:


# book和movie两个domain交集的用户数量
len(set(domain_set[0]["user"]) & set(domain_set[1]["user"])), len(set(domain_set[0]["user"])), len(set(domain_set[1]["user"]))


# In[ ]:


# domain A中物品数量, domian B中物品数量, 用户数量
len(item_dict[0]["str2id"]), len(item_dict[1]["str2id"]), len(user_dict["str2id"])


# In[ ]:


# 验证map是否能对上
print(item_count)
max(item_dict[0]["str2id"].values()), max(item_dict[1]["str2id"].values())


# 把所有数据先存下来
# 可以使用final_domain去进行数据筛选

# In[39]:


with open("./handled/id_map.json", "w") as f:
    json.dump({"user_dict": user_dict, "item_dict": item_dict}, f)
with open("./handled/amazon_all.pkl", "wb") as f:
    pickle.dump((final_data, final_domain), f)


# In[3]:


with open("./handled/id_map.json", "r") as f:
    map_dict = json.load(f)
user_dict = map_dict["user_dict"]
item_dict = map_dict["item_dict"]

with open("./handled/amazon_all.pkl", "rb") as f:
    final_data, final_domain = pickle.load(f)


# 筛选book-movie两个domain
# 
# 这里选的是book和movie两个domain

# In[ ]:


## 先筛选final_data和final_domain
bm_data, bm_domain = {}, {}
for user_id, inter in tqdm(final_domain.items()):
    inter = np.array(inter)
    inter_data = np.array(final_data[user_id])
    bm_data[user_id] = inter_data[np.where(np.logical_or(inter==0, inter == 1))]
    bm_domain[user_id] = inter[np.where(np.logical_or(inter==0, inter == 1))]


# In[42]:


domain_stats = []
for inter in bm_domain.values():
    domain_stats.append(np.mean(inter))


# In[ ]:


# 统计两个domain中overlap的用户
domain_stats = np.array(domain_stats)
domain_stats[domain_stats==0].shape[0], domain_stats[domain_stats==1].shape[0], domain_stats.shape[0]


# In[ ]:


# 统计整体序列的长度
inter_len = []
for inter in bm_data.values():
    inter_len.append(len(inter))
print(np.mean(inter_len))
plt.hist(inter_len, bins=30)


# In[ ]:


min(inter_len)


# In[ ]:


# 统计物品的交互次数
item_freq = {
        0: np.zeros(item_count[0]+1),
        1: np.zeros(item_count[1]+1),
    }
for user_id in tqdm(final_data.keys()):
    seq = final_data[user_id]
    domain_seq = final_domain[user_id]
    for i in range(len(seq)):
        item_freq[domain_seq[i]][seq[i]] += 1


# In[47]:


# 方便画频率分布直方图
item_freq[0][item_freq[0]>30] = 30
item_freq[1][item_freq[1]>30] = 30


# In[ ]:


np.mean(item_freq[0]), np.mean(item_freq[1])


# In[ ]:


plt.hist(item_freq[0], bins=30)


# In[ ]:


plt.hist(item_freq[1], bins=30)


# In[ ]:


inter_len = np.array(inter_len)
len(inter_len[inter_len>200]) / len(inter_len)


# In[52]:


with open("./handled/cloth_sport.pkl", "wb") as f:
    pickle.dump((bm_data, bm_domain), f)


# In[53]:


# 统计重复交互的问题
# _, i_counts = np.unique(bm_data[0], return_counts=True)
# np.sum(i_counts), len(i_counts)


# get attributes

# In[61]:


def get_attribute_Amazon(meta_infos, datamaps, attribute_core):

    attributes = defaultdict(int)
    # 做映射
    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []

    for iid, attributes in meta_infos.items():
        item_id = datamaps['item2id'][iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(f'before delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')
    # 更新datamap
    datamaps['attribute2id'] = attribute2id
    datamaps['id2attribute'] = id2attribute
    datamaps['attributeid2num'] = attributeid2num
    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes


# In[4]:


from data_process import parse_meta
import json


# In[ ]:


import gzip
def parse_meta(path): # for Amazon
    g = gzip.open(path, 'rb')
    inter_list = []
    for l in tqdm(g):
        json_str = l.decode()
        inter_list.append(json.loads(l))

    return inter_list


# In[7]:


def Amazon_meta(dataset_name, data_maps):
    datas = {}
    meta_flie = './raw/meta_' + str(dataset_name) + '.json.gz'
    item_asins = list(data_maps['str2id'].keys())

    for info in tqdm(parse_meta(meta_flie)):
        if info['asin'] not in item_asins:
            continue
        datas[info['asin']] = info
    return datas


# In[ ]:


meta_data_A = Amazon_meta(domain_A, item_dict["0"])
# meta_data_B = Amazon_meta(domain_B, item_dict["1"])


# In[ ]:


len(meta_data_A)


# In[72]:


json_str = json.dumps(meta_data_A)
with open("./handled/item2attributes_A.json", 'w') as out:
    out.write(json_str)


# In[8]:


meta_data_B = Amazon_meta(domain_B, item_dict["1"])


# In[9]:


json_str = json.dumps(meta_data_B)
with open("./handled/item2attributes_B.json", 'w') as out:
    out.write(json_str)


# In[ ]:


meta_data_C = Amazon_meta(domain_C, item_dict["2"])


# In[ ]:


json_str = json.dumps(meta_data_C)
with open("./handled/item2attributes_C.json", 'w') as out:
    out.write(json_str)

