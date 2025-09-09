# 3_analyze_data.py

import numpy as np
from tqdm import tqdm
import pickle
import json
from collections import defaultdict
import os

# --- 設定項目 ---
HANDLED_DIR = "./handled"
# どのドメインペアのデータを最終的に保存するかを指定
# 例: [0, 1] はドメインAとBのペア
# 3ドメイン全てを対象とする場合は、このリストは使用しません
TARGET_DOMAIN_IDS = [0, 1, 2] 
# 保存するファイル名を指定
OUTPUT_FILENAME = "cloth_sport_fashion.pkl" 
# --- ここまで ---

# --- ステップ1: 処理済みファイルを読み込む ---
id_map_path = os.path.join(HANDLED_DIR, "id_map.json")
interactions_path = os.path.join(HANDLED_DIR, "amazon_all.pkl")

if not (os.path.exists(id_map_path) and os.path.exists(interactions_path)):
    print(f"Error: Required files not found in '{HANDLED_DIR}'.")
    print("Please run '1_process_interactions.py' first.")
    exit()

print("--- Loading processed data ---")
with open(id_map_path, "r") as f:
    map_dict = json.load(f)
item_count = map_dict["item_dict"]["item_count"]

with open(interactions_path, "rb") as f:
    final_data, final_domain = pickle.load(f)

num_domains = len(item_count)
print(f"Found {num_domains} domains in the source files.")

# --- ステップ2: 指定されたドメインのデータを抽出 ---
print(f"\n--- Filtering for target domains: {TARGET_DOMAIN_IDS} ---")
filtered_data, filtered_domain = {}, {}
for user_id, domains_list in tqdm(final_domain.items(), desc="Filtering user sequences"):
    # NumPy配列に変換して効率的にフィルタリング
    inter_arr = np.array(final_data[user_id])
    domain_arr = np.array(domains_list)
    
    # ユーザーのシーケンス内にターゲットドメインのアイテムが存在するかチェック
    mask = np.isin(domain_arr, TARGET_DOMAIN_IDS)
    
    # ターゲットドメインのアイテムが1つでもあれば、そのシーケンスを保持
    if np.any(mask):
        filtered_data[user_id] = inter_arr[mask]
        filtered_domain[user_id] = domain_arr[mask]

print(f"Filtered down to {len(filtered_data)} users active in the target domains.")


# --- ステップ3: 3ドメイン対応のデータ統計分析 ---
# ユーザーごとのドメイン利用状況を分析
print("\n--- Analyzing domain statistics per user ---")
user_counts_per_domain = defaultdict(int)
overlap_users = 0
for domains_set in [set(d) for d in filtered_domain.values()]:
    if len(domains_set) == 1:
        domain_id = list(domains_set)[0]
        user_counts_per_domain[domain_id] += 1
    else:
        overlap_users += 1

for i in TARGET_DOMAIN_IDS:
    print(f"Users exclusively in domain {i}: {user_counts_per_domain[i]}")
print(f"Users active in multiple domains: {overlap_users}")

# 全体的なシーケンス長の統計
inter_len = [len(inter) for inter in filtered_data.values()]
print("\n--- Sequence length statistics ---")
if inter_len:
    print(f"Average sequence length: {np.mean(inter_len):.2f}")
    print(f"Min sequence length: {np.min(inter_len)}")
    print(f"Max sequence length: {np.max(inter_len)}")
else:
    print("No sequences to analyze.")

# ドメインごとのアイテム出現頻度を計算
print("\n--- Item frequency statistics per domain ---")
item_freq = {str(i): defaultdict(int) for i in range(num_domains)}
for user_id in tqdm(filtered_data.keys(), desc="Calculating item frequencies"):
    seq = filtered_data[user_id]
    domain_seq = filtered_domain[user_id]
    for i in range(len(seq)):
        item_freq[str(domain_seq[i])][seq[i]] += 1

for i in TARGET_DOMAIN_IDS:
    domain_key = str(i)
    if domain_key in item_freq and item_freq[domain_key]:
        avg_freq = np.mean(list(item_freq[domain_key].values()))
        print(f"Domain {i} - Average item frequency: {avg_freq:.2f}")
    else:
        print(f"Domain {i} - No items to analyze.")


# --- ステップ4: 最終的なデータ構造の保存 ---
output_path = os.path.join(HANDLED_DIR, OUTPUT_FILENAME)
print(f"\n--- Saving final filtered data to {output_path} ---")
with open(output_path, "wb") as f:
    # NumPy配列をリストに戻して保存
    final_filtered_data = {k: v.tolist() for k, v in filtered_data.items()}
    final_filtered_domain = {k: v.tolist() for k, v in filtered_domain.items()}
    pickle.dump((final_filtered_data, final_filtered_domain), f)

print("Analysis and final data saving finished successfully!")