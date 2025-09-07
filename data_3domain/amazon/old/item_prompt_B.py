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


data = json.load(open("./handled/item2attributes_B.json", "r"))


# In[3]:


len(data)


# In[4]:


example_dict = {}
for item_dict in tqdm(data.values()):
    example_dict.update(item_dict)


# In[ ]:


id_map = json.load(open("./handled/id_map.json", "r"))["item_dict"]["1"]["str2id"]
title_data = {}
for key, value in tqdm(data.items()):
    title_data[id_map[key]] = value["title"][:100]


# In[6]:


title_list = []
for id in range(1, len(id_map)+1):
    try:
        title_list.append(title_data[id])
    except:
        title_list.append("no name")

with open("./handled/title_B.pkl", "wb") as f:
    pickle.dump(title_list, f)


# In[ ]:


example_dict.keys()


# In[ ]:


example_dict["main_cat"]


# In[ ]:


example_dict["description"][0]


# In[21]:


def get_attri(item_str, attri, item_info):

    if attri not in item_info.keys() or len(item_info[attri]) > 100:
        new_str = item_str.replace(f"<{attri.upper()}>", "unknown")
    else:
        new_str = item_str.replace(f"<{attri.upper()}>", str(item_info[attri]))

    return new_str


# In[22]:


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


prompt_template = "The sports item has following attributes: \n name is <TITLE>; brand is <BRAND>; price is <PRICE>, rating is <DATE>, price is <PRICE>. \n"
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


# In[ ]:


item_data


# In[ ]:


len_list = []
for item_str in item_data.values():
    len_list.append(len(item_str))
np.mean(len_list)


# In[27]:


json.dump(item_data, open("./handled/item_str_B_truncate.json", "w"))


# In[28]:


# convert to jsonline
def save_data(data_path, data):
    '''write all_data list to a new jsonl'''
    with jsonlines.open("./handled/"+ data_path, "w") as w:
        for meta_data in data:
            w.write(meta_data)

id_map = json.load(open("./handled/id_map.json", "r"))["item_dict"]["1"]["str2id"]
json_data = []
for key, value in item_data.items():
    json_data.append({"input": value, "target": "", "item": key, "item_id": id_map[key]})

json_data = sorted(json_data, key=lambda x: x["item_id"])
save_data("item_str_B_truncate.jsonline", json_data)


# In[29]:


import requests
import json


# In[30]:

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

# In[ ]:


value_list = []

for key, value in tqdm(item_data.items()):
    if len(value) > 4096:
        value_list.append(key)


#
# 埋め込みキャッシュファイルを読み込む（存在しない場合は空の辞書を作成）
if os.path.exists("./handled/item_emb_A.pkl"):
    item_emb = pickle.load(open("./handled/item_emb_A.pkl", "rb"))
else:
    item_emb = {}

# 1. すでに処理済みのアイテムのキーをsetとして取得（検索が高速になります）
processed_keys = set(item_emb.keys())

# 2. これから処理すべきアイテムのキーリストを作成
#    item_dataのキーのうち、まだ処理されていないものだけを抽出します
keys_to_process = [key for key in item_data.keys() if key not in processed_keys]

# 処理状況を表示
print(f"Total items in item_data: {len(item_data)}")
print(f"Already processed items: {len(processed_keys)}")
print(f"Items remaining to process: {len(keys_to_process)}")

# 3. 処理すべきアイテムが残っている場合のみループを実行
if keys_to_process:
    try:
        # 進捗バー(tqdm)で、処理対象のアイテムリストをループします
        with tqdm(total=len(keys_to_process), desc="Generating Embeddings") as pbar:
            for i, key in enumerate(keys_to_process):
                # item_dataから対象のテキストを取得
                value = item_data[key]
                if len(value) > 4096:
                    value = value[:4095]
                
                # 埋め込みを取得して辞書に保存
                item_emb[key] = get_response(value)
                
                # 進捗バーを1つ進める
                pbar.update(1)

                # 安全のため、100件処理するごとに進捗をファイルに保存します
                if (i + 1) % 100 == 0:
                    pickle.dump(item_emb, open("./handled/item_emb_A.pkl", "wb"))

    except Exception as e:
        # APIエラーなど、何か問題が発生した場合の処理
        print(f"\nAn error occurred: {e}")
        print("Saving current progress before exiting...")
    
    finally:
        # エラーが発生した場合でも、正常に完了した場合でも、
        # 最後に必ず最新の進捗をファイルに保存します
        print("\nSaving final progress...")
        pickle.dump(item_emb, open("./handled/item_emb_A.pkl", "wb"))
        print("Progress saved successfully.")
else:
    print("All items have already been processed. Nothing to do.")

# In[47]:


id_map = json.load(open("./handled/id_map.json", "r"))["item_dict"]["1"]["id2str"]
emb_list = []
for id in range(1, len(id_map)+1):
    try:    # 有一个物品没有属性，给其赋0向量
        meta_emb = item_emb[id_map[str(id)]]
    except:
        meta_emb = [0] * len(list(item_emb.values())[0])
    emb_list.append(meta_emb)

emb_list = np.array(emb_list)
pickle.dump(emb_list, open("./handled/itm_emb_np_B.pkl", "wb"))


# In[ ]:


# 确保LLM embedding和物品的数量是相同的
assert len(emb_list) == len(id_map)

