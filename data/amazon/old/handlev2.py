#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import json
from data_process import New_Amazon, Amazon_meta
from collections import defaultdict
import os

# --- ステップ1: 各ドメインの生データを処理し、中間ファイルとして保存 ---

# 中間ファイルを保存するディレクトリを定義
PROCESSED_DATA_DIR = "./processed_data"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# 処理対象のドメインを定義（今後、ここに追加するだけでOKです）
domains = {
    0: "Clothing_Shoes_and_Jewelry",
    1: "Sports_and_Outdoors",
    2: "AMAZON_FASHION"
}

# 各ドメインの生データを読み込み、ドメインIDを付与して中間ファイルに保存
for domain_id, domain_name in domains.items():
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, f'data_{domain_name}_processed.pkl')
    
    if not os.path.exists(processed_file_path):
        print(f"--- Processing raw data for domain '{domain_name}' ---")
        
        # New_Amazonジェネレータからデータを1つずつ処理
        processed_data = [
            list(inter) + [domain_id]
            for inter in tqdm(New_Amazon(domain_name, 0), desc=f"Processing {domain_name}")
        ]
            
        with open(processed_file_path, 'wb') as f:
            pickle.dump(processed_data, f)
        print(f"Saved processed data for '{domain_name}' to {processed_file_path}")
    else:
        print(f"--- Found existing processed file for '{domain_name}', skipping raw data processing ---")

# --- ステップ2: 保存した全ての中間ファイルを読み込んで結合 ---

all_data = []
print("\n--- Loading and merging all processed data ---")
for domain_id, domain_name in domains.items():
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, f'data_{domain_name}_processed.pkl')
    print(f"Loading {processed_file_path}...")
    with open(processed_file_path, 'rb') as f:
        data = pickle.load(f)
        all_data.extend(data)

print(f"\nTotal interactions from all domains: {len(all_data)}")


# --- ステップ3: 結合後のデータに対してフィルタリング等の処理を実行 ---
# (ここから先の関数定義は、元のスクリプトからコピーしてください)

def check_Kcore(data, user_core, item_core):
    user_count, item_count = defaultdict(int), defaultdict(int)
    for inter in data:
        user_count[inter[0]] += 1
        item_count[inter[1]] += 1
    
    user_check = all(count >= user_core for count in user_count.values())
    item_check = all(count >= item_core for count in item_count.values())
    
    return user_count, item_count, user_check and item_check

def filter_Kcore(data, user_core, item_core):
    while True:
        user_count, item_count = defaultdict(int), defaultdict(int)
        for inter in data:
            user_count[inter[0]] += 1
            item_count[inter[1]] += 1
        
        core_data = [
            inter for inter in data 
            if user_count[inter[0]] >= user_core and item_count[inter[1]] >= item_core
        ]
        
        if len(core_data) == len(data):
            print("K-core filter done!")
            domain_set = {i: {"user": [], "item": []} for i in range(len(domains))}
            for inter in core_data:
                user_id, item_id, _, domain_id = inter
                domain_set[domain_id]["user"].append(user_id)
                domain_set[domain_id]["item"].append(item_id)
            return core_data, domain_set
        data = core_data

def filter_time(data, t_min, t_max):
    print("Filtering by time...")
    return [inter for inter in tqdm(data) if t_min < inter[2] < t_max]

def id_map(data, domain_set):
    final_data, final_domain = {}, {}
    temp_data = {}
    new_user_id = 1
    item_count = {i: 1 for i in range(len(domains))}
    item_dict = {i: {"str2id": {}, "id2str": {}} for i in range(len(domains))}
    user_dict = {"str2id": {}, "id2str": {}}
    for inter in tqdm(data, desc="Mapping IDs"):
        user_id, item_id, time, domain_id = inter
        if item_id not in item_dict[domain_id]["str2id"]:
            new_item_id = item_count[domain_id]
            item_dict[domain_id]["str2id"][item_id] = new_item_id
            item_dict[domain_id]["id2str"][new_item_id] = item_id
            item_count[domain_id] += 1
        if user_id not in user_dict["str2id"]:
            user_dict["str2id"][user_id] = new_user_id
            user_dict["id2str"][new_user_id] = user_id
            temp_data[new_user_id] = []
        temp_data[new_user_id].append((item_dict[domain_id]["str2id"][item_id], domain_id, time))
    print("Map done!")
    for user_id, inter in tqdm(temp_data.items(), desc="Sorting sequences"):
        inter.sort(key=lambda x: x[2])
        final_data[user_id] = [t[0] for t in inter]
        final_domain[user_id] = [t[1] for t in inter]
    print("Sort done!")
    return final_data, final_domain, user_dict, item_dict, item_count

print("\n--- Starting final filtering and mapping process ---")
all_data = filter_time(all_data, t_min=1514736000, t_max=1577808000)
new_data, domain_set = filter_Kcore(all_data, user_core=5, item_core=3)
final_data, final_domain, user_dict, item_dict, _ = id_map(new_data, domain_set)
item_count = {domain_id: len(set(domain_set[domain_id]["item"])) for domain_id in domain_set.keys()}
item_dict["item_count"] = item_count

# --- ステップ4: 最終結果の保存 ---
print("\n--- Saving final processed files ---")
with open("./handled/id_map.json", "w") as f:
    json.dump({"user_dict": user_dict, "item_dict": item_dict}, f)
with open("./handled/amazon_all.pkl", "wb") as f:
    pickle.dump((final_data, final_domain), f)

print("All processing finished successfully!")