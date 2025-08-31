# 2_fetch_metadata.py

import json
import os
from data_process import Amazon_meta
from collections import defaultdict

# --- 設定項目 ---
HANDLED_DIR = "./handled"
domains = {
    0: "Clothing_Shoes_and_Jewelry",
    1: "Sports_and_Outdoors",
    2: "AMAZON_FASHION"
}
# --- ここまで ---

# --- ステップ1: id_map.jsonを読み込む ---
id_map_path = os.path.join(HANDLED_DIR, "id_map.json")
if not os.path.exists(id_map_path):
    print(f"Error: '{id_map_path}' not found. Please run '1_process_interactions.py' first.")
    exit()

with open(id_map_path, "r") as f:
    map_dict = json.load(f)
item_dict = map_dict["item_dict"]

# --- ステップ2: 各ドメインのメタデータを取得・保存 ---
print("\n--- Fetching and saving metadata for filtered items ---")
for domain_id, domain_name in domains.items():
    meta_file_path = os.path.join(HANDLED_DIR, f"item2attributes_{chr(ord('A') + domain_id)}.json")
    
    if not os.path.exists(meta_file_path):
        print(f"Fetching metadata for domain '{domain_name}'...")
        if str(domain_id) in item_dict:
            # フィルタリング後のitem IDのみを対象にメタデータを取得
            meta_data = Amazon_meta(domain_name, item_dict[str(domain_id)])
            with open(meta_file_path, 'w') as out:
                json.dump(meta_data, out)
            print(f"Saved metadata to {meta_file_path}")
        else:
            print(f"No items left for domain '{domain_name}' after filtering, skipping metadata.")
    else:
        print(f"--- Found existing metadata file for '{domain_name}', skipping ---")

print("\nMetadata fetching finished successfully!")