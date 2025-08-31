#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import jsonlines
import pandas as pd
import numpy as np
import json
import copy
from tqdm import tqdm
import shutil

# In[2]:


data = json.load(open("./handled/item2attributes_A.json", "r"))


# In[ ]:


len(data)


# In[ ]:


example_dict = {}
for item_dict in tqdm(data.values()):
    example_dict.update(item_dict)


# In[ ]:


id_map = json.load(open("./handled/id_map.json", "r"))["item_dict"]["0"]["str2id"]
title_data = {}
for key, value in tqdm(data.items()):
    title_data[id_map[key]] = value["title"][:100]


# In[ ]:


# the number of items that do not have name
print("the number of items that do not have name: {}".format(len(id_map.values()) - len(data)))


# In[ ]:


title_list = []
for id in range(1, len(id_map)+1):
    if id not in title_data.keys():
        title_list.append("no name")
    else:
        title_list.append(title_data[id])

assert len(title_list) == len(id_map)

with open("./handled/title_A.pkl", "wb") as f:
    pickle.dump(title_list, f)


# In[ ]:


example_dict.keys()


# In[ ]:


example_dict["description"][0]


# In[12]:


def get_attri(item_str, attri, item_info):

    if attri not in item_info.keys() or len(item_info[attri]) > 100:
        new_str = item_str.replace(f"<{attri.upper()}>", "unknown")
    else:
        new_str = item_str.replace(f"<{attri.upper()}>", str(item_info[attri]))

    return new_str


# In[13]:


def get_feat(item_str, feat, item_info):

    if feat not in item_info.keys():
        return ""
    
    assert isinstance(item_info[feat], list)
    feat_str = ""
    for meta_feat in item_info[feat]:
        feat_str = feat_str + meta_feat + "; "
    new_str = item_str.replace(f"<{feat.upper()}>", feat_str)

    if len(new_str) > 128: # avoid exceed the input length limitation
        return new_str[:128]

    return new_str


# In[ ]:


prompt_template = "The electronic item has following attributes: \n name is <TITLE>; brand is <BRAND>; price is <PRICE>, rating is <DATE>, price is <PRICE>. \n"
feat_template = "The item has following features: <CATEGORY>. \n"
desc_template = "The item has following descriptions: <DESCRIPTION>. \n"


# In[ ]:


item_data = {}
for key, value in tqdm(data.items()):
    item_str = copy.deepcopy(prompt_template)
    item_str = get_attri(item_str, "title", value)
    item_str = get_attri(item_str, "brand", value)
    item_str = get_attri(item_str, "date", value)
    # item_str = get_attri(item_str, "rank", value)
    item_str = get_attri(item_str, "price", value)

    feat_str = copy.deepcopy(feat_template)
    feat_str = get_feat(feat_str, "category", value)
    desc_str = copy.deepcopy(desc_template)
    desc_str = get_feat(desc_str, "description", value)
    
    item_data[key] = item_str + feat_str + desc_str


# In[16]:


len_list = []
for item_str in item_data.values():
    len_list.append(len(item_str))


# In[ ]:


np.mean(len_list)


# In[18]:


json.dump(item_data, open("./handled/item_str_A_truncate.json", "w"))


# In[19]:


# convert to jsonline
def save_data(data_path, data):
    '''write all_data list to a new jsonl'''
    with jsonlines.open("./handled/"+ data_path, "w") as w:
        for meta_data in data:
            w.write(meta_data)

id_map = json.load(open("./handled/id_map.json", "r"))["item_dict"]["0"]["str2id"]
json_data = []
for key, value in item_data.items():
    json_data.append({"input": value, "target": "", "item": key, "item_id": id_map[key]})

json_data = sorted(json_data, key=lambda x: x["item_id"])
save_data("item_str_A_truncate.jsonline", json_data)


# In[20]:


import requests
import json


# In[21]:



def get_response(prompt):
    url = "https://api.openai.com/v1/embeddings"

    payload = json.dumps({
    "model": "text-embedding-ada-002",
    "input": prompt
    })
    api_key = "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"
    headers = {
    'Authorization': f'Bearer {api_key}',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
    }

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        
        # ステータスコードが4xxや5xx（エラー）の場合、ここで例外を発生させる
        response.raise_for_status() 
        
        re_json = response.json()

        # 応答に 'data' キーが含まれているか安全にチェック
        if "data" in re_json and len(re_json["data"]) > 0:
            return re_json["data"][0]["embedding"]
        else:
            # 'data' がない、または空の場合はエラーとして扱う
            print("Error: API response did not contain expected data.")
            print("Full response:", re_json)
            raise ValueError("Invalid API response format")

    except requests.exceptions.HTTPError as http_err:
        # HTTPエラー（認証エラー、レート制限など）をここで捕捉
        print(f"HTTP error occurred: {http_err}")
        print("Response body:", response.text)
        raise # エラーを再発生させてプログラムを安全に停止

    except Exception as e:
        # その他の予期せぬエラー（ネットワーク接続の問題など）
        print(f"An unexpected error occurred: {e}")
        raise

# In[22]:


item_emb = {}
filepath = "./handled/item_emb_A.pkl"
temp_filepath = filepath + ".tmp" # 一時保存用のファイルパスを定義  <-- 変更点

# ファイルが存在する場合のみ読み込みを試みる
if os.path.exists(filepath):
    try:
        # ファイルサイズが0より大きいことを確認してから開く
        if os.path.getsize(filepath) > 0:
            with open(filepath, "rb") as f:
                item_emb = pickle.load(f)
        else:
            print(f"Warning: Cache file '{filepath}' is empty. Deleting it.")
            os.remove(filepath) # 空のファイルは削除
            
    except EOFError:
        # ファイルが破損している場合
        print(f"Warning: Cache file '{filepath}' is corrupt. Deleting it.")
        os.remove(filepath) # 破損したファイルは削除
        item_emb = {}

# 1. すでに処理済みのアイテムのキーをsetとして取得
processed_keys = set(item_emb.keys())

# 2. これから処理すべきアイテムのキーリストを作成
keys_to_process = [key for key in item_data.keys() if key not in processed_keys]

# 処理状況を表示
print(f"Total items in item_data: {len(item_data)}")
print(f"Already processed items: {len(processed_keys)}")
print(f"Items remaining to process: {len(keys_to_process)}")

# 3. 処理すべきアイテムが残っている場合のみループを実行
if keys_to_process:
    try:
        with tqdm(total=len(keys_to_process), desc="Generating Embeddings") as pbar:
            for i, key in enumerate(keys_to_process):
                value = item_data[key]
                if len(value) > 4096:
                    value = value[:4095]
                
                item_emb[key] = get_response(value) # get_response関数を呼び出し
                
                pbar.update(1)

                # --- 安全な定期保存 ---
                if (i + 1) % 100 == 0:
                    # 1. 一時ファイルに書き込む
                    with open(temp_filepath, "wb") as f:
                        pickle.dump(item_emb, f)
                    # 2. 成功したら、元のファイルにリネーム（置き換え）
                    shutil.move(temp_filepath, filepath)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Saving current progress before exiting...")
    
    finally:
        # --- 安全な最終保存 ---
        print("\nSaving final progress...")
        try:
            with open(temp_filepath, "wb") as f:
                pickle.dump(item_emb, f)
            shutil.move(temp_filepath, filepath)
            print("Progress saved successfully.")
        except Exception as save_err:
            print(f"Failed to save final progress: {save_err}")

else:
    print("All items have already been processed. Nothing to do.")

# In[ ]:


id_map = json.load(open("./handled/id_map.json", "r"))["item_dict"]["0"]["id2str"]
emb_list = []
for id in range(1, len(id_map)+1):
    if id_map[str(id)] in item_emb.keys():
        meta_emb = item_emb[id_map[str(id)]]
    else:
        meta_emb = [0] * len(list(item_emb.values())[0])
    emb_list.append(meta_emb)

emb_list = np.array(emb_list)
pickle.dump(emb_list, open("./handled/itm_emb_np_A.pkl", "wb"))


# In[ ]:


# 确保LLM embedding和物品的数量是相同的
assert len(emb_list) == len(id_map)

