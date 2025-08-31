
import os
import numpy as np
from tqdm import tqdm
import pickle
import json
from data_process import New_Amazon, Amazon_meta
from collections import defaultdict

import config

class AmazonHandler():
    def __init__(self):
        self.domains_map = config.DOMAINS
        self.user_core = config.USER_CORE
        self.item_core = config.ITEM_CORE
        
        # クラス内で使用する変数をすべて初期化
        self.data = []
        self.new_data = []
        self.user_count = {}
        self.item_count = {}
        self.domain_set = {int(key): {"user": set(), "item": set()} for key in self.domains_map.keys()}
        self.final_data = {}
        self.final_domain = {}
        self.user_dict = {}
        self.item_dict = {}

        os.makedirs(config.HANDLE_DATA_DIR, exist_ok=True)

    def run_pipeline(self):
        print("--- 1.1 : Loading and Combining Data ---")
        self.load_data()
        
        print("--- 1.2 : Filtering Data by Time ---")
        self.filter_time(t_min=config.TIME_MIN, t_max = config.TIME_MAX)

        print("--- 1.3 : Counting Interactions ---")
        self.id_map()

        print("--- Step 5 : Saving Data ---")

        with open(f"{config.HANDLE_DATA_DIR}/id_map.json", "w") as f:
            all_maps = {"user_dict":self.user_dict, "item_dict":self.item_dict}
            json.dump(all_maps, f, indent=4)
            print(f"Saved id_map.json")
        
        with open(f"{config.HANDLE_DATA_DIR}/amazon_all.pkl", "wb") as f:
            pickle.dump((self.final_data, self.final_domain), f)
            print(f"Saved amazon_all.pkl")
        
        self.export_sequences_for_domains(
            domain_ids=[0, 1, 2], 
            output_path=f"{config.HANDLE_DATA_DIR}/cloth_sport_fashion.pkl"
        )

        self.fetch_metadata(
            domain_ids=[0, 1, 2],
            output_path=f"{config.HANDLE_DATA_DIR}/"
        )

        print("\n--- Processing Summary ---")

    def load_data(self):
        all_data = []
        for domain_id_str, domain_name in self.domains_map.items():
            try:
                domain_data = New_Amazon(domain_name, 0)
            except:
                print(f"Error loading data for domain {domain_name}")
                continue
            
            for inter in tqdm(domain_data, desc="Loading '{domain_name}'"):
                new_inter = list(inter)
                new_inter.append(int(domain_id_str))
                all_data.append(new_inter)
        
        self.data = all_data
        print(f"Data loaded. Total:{len(self.data)} interactions")

    def filter_time(self,t_min=1451577600, t_max=1459440000):   # 过滤掉交互少的数据
        new_data = []

        for inter in tqdm(self.data):
            _, _, time, _ = inter
            if time > t_min and time < t_max:    # 只取2016-01-01到2016-01-15之间的数据
                self.new_data.append(inter)

        self.data = new_data

        print(f"Time filter done!. Interactions remaining: { len(self.data) }")        
    
    def count_inter(self,t_min,t_max):
        for inter in self.data:
            user_id, item_id, time, _ = inter
            
            if user_id not in self.user_count.keys():
                self.user_count[user_id] = 1
            else:
                if time > t_min and time < t_max:
                    self.user_count[user_id] += 1

            if item_id not in self.item_count.keys():
                self.item_count[item_id] = 1
            else:
                if time > t_min and time < t_max:
                    self.item_count[item_id] += 1
        return self.user_count, self.item_count

    def filter(self,user_minmum, item_minimum, t_min=1451577600, t_max=1459440000):   # 过滤掉交互少的数据

        for inter in tqdm(self.data):
            user_id, item_id, time, domain_id = inter
            
            if self.item_count[item_id] > item_minimum and self.user_count[user_id] > user_minmum \
            and time > t_min and time < t_max:    # 只取2016-01-01到2016-01-15之间的数据
                self.new_data.append(inter)
                self.domain_set[domain_id]["user"].append(user_id)
                self.domain_set[domain_id]["item"].append(item_id)
        print("filter done!")

        return self.new_data, self.domain_set

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

    # K-core user_core item_core
    @staticmethod
    def check_Kcore(data, user_core, item_core):
        user_count = defaultdict(int)
        item_count = defaultdict(int)
        
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
    def filter_Kcore(self, user_core, item_core): # user 接所有items
        
        
        while True:
            user_count, item_count , is_kcore = self.check_Kcore(self.data, user_core, item_core)
            
            if is_kcore:
                print("K-core condition met.")
                break

            self.data = [
                inter for inter in tqdm(self.data, desc="Filtering K-core")
                if user_count[inter[0]] >= user_core and item_count[inter[1]] >= item_core
            ]
            print(f"K-core filter done!. Interactions remaining: { len(self.data) }")

        final_domain_set = {i : {"user": set(), "item": set()} for i in range(len(self.domains))}
        for user_id, item_id, _, domain_in in self.data:
            final_domain_set[domain_id]["user"].add(user_id)
            final_domain_set[domain_id]["item"].add(item_id)
        
        self.domain_set = final_domain_set
        print(f"K-core filter complete.\n")
    
    def id_map(self):
        print("--- Step 4 : Mapping IDs and Building Sequences ---")
        temp_data = defaultdict(list)
        new_user_id = 1

        self.user_dict = {"str2id": {}, "id2str": {}}
        self.item_dict = {
            i:{"str2id" : {} , "id2str" : {}} for i in range(len(self.domains))
        }
        item_count = defaultdict(lambda: 1)

        for user_id, item_id, time, domain_id in tqdm(self.data, desc="Mapping IDs"):
            if item_id not in self.item_dict[domain_id]["str2id"]:
                new_item_id = item_count[domain_id]
                self.item_dict[domain_id]["str2id"][item_id] = new_item_id
                self.item_dict[domain_id]["id2str"][new_item_id] = item_id
                item_count[domain_id] += 1
            
            if user_id not in self.user_data["str2id"]:
                self.user_data["str2id"][user_id] = new_user_id
                self.user_data["id2str"][new_user_id] = user_id
                new_user_id += 1
            
            mapped_user_id = self.user_data["str2id"][user_id]
            mapped_item_id = self.item_dict[domain_id]["str2id"][item_id]
            temp_data[mapped_user_id].append((mapped_item_id, domain_id, time))

        print("ID mapping done!")

        for user_id, inter in tqdm(temp_data.items(),desc = "Sorting sequences"):
            inter.sort(key=lambda x: x[2])
            self.final_data[user_id] = [temp_tuple[0] for temp_tuple in inter]
            self.final_domain[user_id] = [temp_tuple[1] for temp_tuple in inter]
        
        final_item_count = {domain_id : len(items["str2id"] for domain_id, items in self.item_dict.items())}
        self.item_dict["item_count"] = final_item_count
        print("Sort done!")
    
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
            
        print(f"Export complete. {len(filtered_data)} users have sequences with the specified domains."))
        with open(output_path, "wb") as f:
            pickle.dump((filtered_data, filtered_domain), f)
        print(f"Saved {output_path}")

    def fetch_metadata(self,domain_ids: list, output_path: str):
        print(f"--- Fetching metadata for domains {domain_ids} ---")
        for domain_id in domain_ids:
            meta_data = Amazon_meta(self.domains[domain_id], self.item_dict[domain_id])
            json_str = json.dumps(meta_data)
            with open(f"`{output_path}item2attributes_{domain}.json", 'w') as out:
                out.write(json_str)
            return
    

