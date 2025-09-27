import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import json
from data_process import New_Amazon, Amazon_meta
from collections import defaultdict

domain_A = "Clothing_Shoes_and_Jewelry"
domain_B = "Sports_and_Outdoors"
data_A = New_Amazon(domain_A, 0)
data_B = New_Amazon(domain_B, 0)

# 给每个交互添加domain id
new_data_A, new_data_B = [], []
for inter in tqdm(data_A):
    new_inter = list(inter)
    new_inter.append(0)
    new_data_A.append(new_inter)
for inter in tqdm(data_B):
    new_inter = list(inter)
    new_inter.append(1)
    new_data_B.append(new_inter)

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

def filter(data, user_minmum, item_minimum, t_min=1451577600, t_max=1459440000):   # 过滤掉交互少的数据
    
    user_count, item_count = count_inter(data, t_min=t_min, t_max=t_max)
    domain_set = {0: {"user": [], "item": []},
                  1: {"user": [], "item": []},
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

def id_map(data, domain_set):
    
    final_data, final_domain = {}, {}
    temp_data = {}
    new_user_id = 1
    temp_item_count = {domain_id: len(set(domain_set[domain_id]["item"])) for domain_id in domain_set.keys()}
    item_count = {0: 1, 1: 1}
    item_dict = {
        0: {"str2id": {}, "id2str": {},},
        1: {"str2id": {}, "id2str": {},},
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

# 循环过滤 K-core
def filter_Kcore(data, user_core, item_core): # user 接所有items
    
    user_count, item_count, isKcore = check_Kcore(data, user_core, item_core)
    
    new_data = data

    while not isKcore:

        temp_data = []
        domain_set = {
            0: {"user": [], "item": []},
            1: {"user": [], "item": []},
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

def filter_time(data, t_min=1451577600, t_max=1459440000):   # 过滤掉交互少的数据
    

    new_data = []

    for inter in tqdm(data):
        _, _, time, _ = inter
        
        if time > t_min and time < t_max:    # 只取2016-01-01到2016-01-15之间的数据
            
            new_data.append(inter)

    print("filter time done!")

    return new_data
all_data = new_data_A + new_data_B
print(len(all_data))
# new_data, domain_set = filter(all_data, user_minmum=10, item_minimum=10)
all_data = filter_time(all_data, t_min=1514736000, t_max=1577808000)
print(len(all_data))
new_data, domain_set = filter_Kcore(all_data, user_core=5, item_core=3)
print(len(new_data))
final_data, final_domain, user_dict, item_dict, item_count = id_map(new_data, domain_set)
item_count = {domain_id: len(set(domain_set[domain_id]["item"])) for domain_id in domain_set.keys()}
item_dict["item_count"] = item_count

with open("./handled/id_map.json", "w") as f:
    json.dump({"user_dict": user_dict, "item_dict": item_dict}, f)
with open("./handled/amazon_all.pkl", "wb") as f:
    pickle.dump((final_data, final_domain), f)

with open("./handled/id_map.json", "r") as f:
    map_dict = json.load(f)
user_dict = map_dict["user_dict"]
item_dict = map_dict["item_dict"]

with open("./handled/amazon_all.pkl", "rb") as f:
    final_data, final_domain = pickle.load(f)

## 先筛选final_data和final_domain
bm_data, bm_domain = {}, {}
for user_id, inter in tqdm(final_domain.items()):
    inter = np.array(inter)
    inter_data = np.array(final_data[user_id])
    bm_data[user_id] = inter_data[np.where(np.logical_or(inter==0, inter == 1))]
    bm_domain[user_id] = inter[np.where(np.logical_or(inter==0, inter == 1))]

with open("./handled/cloth_sport.pkl", "wb") as f:
    pickle.dump((bm_data, bm_domain), f)

meta_data_A = Amazon_meta(domain_A, item_dict["0"])
json_str = json.dumps(meta_data_A)
with open("./handled/item2attributes_A.json", 'w') as out:
    out.write(json_str)
meta_data_B = Amazon_meta(domain_B, item_dict["1"])
json_str = json.dumps(meta_data_B)
with open("./handled/item2attributes_B.json", 'w') as out:
    out.write(json_str)