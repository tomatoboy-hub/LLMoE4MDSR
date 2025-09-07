
import os
import numpy as np
from tqdm import tqdm
import pickle
import json
from data_process import New_Amazon, Amazon_meta, stream_amazon_reviews_to_parquet
from collections import defaultdict
import pandas as pd
import config

class AmazonHandler():
    def __init__(self):
        self.domains_map = config.DOMAINS
        self.user_core = config.USER_CORE
        self.item_core = config.ITEM_CORE

        self.df = pd.DataFrame()
        # クラス内で使用する変数をすべて初期化
        self.final_data = {}
        self.final_domain = {}
        self.user_dict = {}
        self.item_dict = {}

        os.makedirs(config.HANDLE_DATA_DIR, exist_ok=True)

    def run_pipeline(self):
        print("--- 1.1 : Loading and Combining Data ---")
        #self.load_data()
        self._load_and_combine_data()

        print("--- 1.2 : Filtering Data by Time ---")
        #self.filter_time(t_min=config.TIME_MIN, t_max = config.TIME_MAX)

        self._filter_time(t_min=config.TIME_MIN, t_max = config.TIME_MAX)

        #self.filter_Kcore(user_core=self.user_core, item_core=self.item_core)
        self._filter_Kcore()

        print("--- 1.3 : Counting Interactions ---")
        #self.id_map()
        self._id_map()

        print("--- Step 5 : Saving Data ---")

        # with open(f"{config.HANDLE_DATA_DIR}/id_map.json", "w") as f:
        #     all_maps = {"user_dict":self.user_dict, "item_dict":self.item_dict}
        #     json.dump(all_maps, f, indent=4)
        #     print(f"Saved id_map.json")
        
        # with open(f"{config.HANDLE_DATA_DIR}/amazon_all.pkl", "wb") as f:
        #     pickle.dump((self.final_data, self.final_domain), f)
        #     print(f"Saved amazon_all.pkl")

        self._save_results()

        self.export_sequences_for_domains(
            domain_ids=[0, 1, 2], 
            output_path=f"{config.HANDLE_DATA_DIR}/cloth_sport_fashion.pkl"
        )

        # self.fetch_metadata(
        #     domain_ids=[0, 1, 2],
        #     output_path=f"{config.HANDLE_DATA_DIR}/"
        # )

        print("\n--- Processing Summary ---")

    def _load_and_combine_data(self):
        all_dfs = []
        for domein_id_str, domain_name in self.domains_map.items():
            parquet_path = f'./raw/{domain_name}.parquet'
            if not os.path.exists(parquet_path):
                print(f"Parquet file not found for domain {domain_name}")
                stream_amazon_reviews_to_parquet(domain_name,rating_score=0)
            else:
                print("Found parquet file for domain {domain_name}")
            
            df_domain = pd.read_parquet(parquet_path)
            df_domain['domain_id'] = int(domein_id_str)
            all_dfs.append(df_domain)
        
        self.df = pd.concat(all_dfs, ignore_index=True)
        self.df["time"] = pd.to_numeric(self.df["time"]).astype(np.int64)
        self.df["domain_id"] = self.df["domain_id"].astype(np.int8)
        print(f"Data loaded. Total:{len(self.df)} interactions")

    def _filter_time(self, t_min, t_max):
        initial_count = len(self.df)
        self.df = self.df[(self.df["time"] > t_min) & (self.df["time"] < t_max)]
        print(f"Time filter done!. Interactions remaining: { len(self.df) }")    
    
    def _filter_Kcore(self):
        while True:
            initial_count = len(self.df)
            user_counts = self.df.groupby('user')['user'].transform('size')
            item_counts = self.df.groupby('item')['item'].transform('size')

            mask = (user_counts >= self.user_core) & (item_counts >= self.item_core)
            self.df = self.df[mask]

            print(f"Filtering iteration ... Interactions remaining: { len(self.df) }")
            if len(self.df) == initial_count:
                print(" K-core condition met.")
                break
        print(f"K-core filter complete.\n")
    
    def _id_map(self):
        if self.df.empty:
            print("No data to map.")
            return
        unique_users = self.df['user'].unique()
        self.user_dict = {
        "str2id": {user_str: idx+1 for idx, user_str in enumerate(unique_users)},
        "id2str": {idx + 1 : user_str for idx, user_str in enumerate(unique_users)}
        }

        self.item_dict = {int(key): {"str2id": {}, "id2str": {}} for key in self.domains_map.keys()}
        for domain_id, group in self.df.groupby('domain_id'):
            unique_items = group['item'].unique()
            self.item_dict[domain_id] = {
                'str2id': {item_str: i + 1 for i, item_str in enumerate(unique_items)},
                'id2str': {i + 1: item_str for i, item_str in enumerate(unique_items)}
            }
 
        self.df['new_user_id'] = self.df['user'].map(self.user_dict['str2id'])
        self.df['new_item_id'] = self.df.apply(
            lambda row: self.item_dict[row['domain_id']]['str2id'][row['item']], axis=1)

        
        sorted_df = self.df.sort_values(by=['new_user_id', 'time'])
        grouped = sorted_df.groupby('new_user_id')
        for user_id, group in tqdm(grouped, desc="    Building final sequences"):
            self.final_data[user_id] = group['new_item_id'].tolist()
            self.final_domain[user_id] = group['domain_id'].tolist()

        self.item_dict["item_count"] = {key: len(val["str2id"]) for key, val in self.item_dict.items() if isinstance(val, dict)}
        print("  Sequence building complete.")

    def _save_results(self):
        id_map_path = os.path.join(config.HANDLE_DATA_DIR, "id_map.json")
        item_dict_str_keys = {str(k): v for k, v in self.item_dict.items()}
        with open(id_map_path, "w") as f:
            json.dump({"user_dict":self.user_dict, "item_dict":item_dict_str_keys}, f, indent=4)
        
        inter_path = os.path.join(config.HANDLE_DATA_DIR, "amazon_all.pkl")
        with open(inter_path, "wb") as f:
            pickle.dump((self.final_data, self.final_domain), f)
    

    def export_sequences_for_domains(self,domain_ids: list, output_path: str):
        print(f"--- Exporting sequences for domains {domain_ids} --- ")
        if not self.final_data: 
            print("No sequences available. Please run id_map() first.")
            return
        
        filtered_data = {}
        filtered_domain = {}
        for user_id, domain_seq in tqdm(self.final_domain.items(), desc="Filetering sequences"):
            domain_seq_np = np.array(domain_seq)
            item_seq_np = np.array(self.final_data[user_id])

            mask = np.isin(domain_seq_np, domain_ids)
            if np.any(mask):
                filtered_data[user_id] = item_seq_np[mask].tolist()
                filtered_domain[user_id] = domain_seq_np[mask].tolist()
            
        print(f"Export complete. {len(filtered_data)} users have sequences with the specified domains.")
        with open(output_path, "wb") as f:
            pickle.dump((filtered_data, filtered_domain), f)
        print(f"Saved {output_path}")

    def fetch_metadata(self,domain_ids: list, output_path: str):
        print(f"--- Fetching metadata for domains {domain_ids} ---")
        for domain_id in domain_ids:
            meta_data = Amazon_meta(self.domains_map[str(domain_id)], self.item_dict[domain_id])
            json_str = json.dumps(meta_data)

            with open(f"`{output_path}item2attributes_{domain_id}.json", 'w') as out:
                out.write(json_str)
        return
    

