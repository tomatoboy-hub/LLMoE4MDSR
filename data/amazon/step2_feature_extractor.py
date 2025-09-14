import os 
import json
import pickle
import numpy as np 
import pandas as pd
from tqdm import tqdm
import copy 
import config
from llm_services import LocalEmbedder
from data_process import stream_amazon_meta_to_parquet

class ItemFeatureExtractor:
    def __init__(self, embedder: LocalEmbedder):
        self.embedder = embedder
        self.id_map_full = json.load(open(os.path.join(config.HANDLE_DATA_DIR, "id_map.json"), "r"))

    def _build_prompts_from_parquet(self, meta_df:pd.DataFrame,formatted_template:str):
        item_prompts = {}
        for index, row in tqdm(meta_df.iterrows(),total=len(meta_df),desc="Building prompts from parquet"):
            item_str = copy.deepcopy(formatted_template)
            # .get(key, default) の代わりに、辞書ライクなアクセスが可能
            item_str = item_str.replace("<TITLE>", str(row.get('title', 'unknown'))[:100])
            item_str = item_str.replace("<BRAND>", str(row.get('brand', 'unknown')))
            item_str = item_str.replace("<PRICE>", str(row.get('price', 'unknown')))
            item_prompts[row['asin']] = item_str
        return item_prompts
        
    def run_for_domain(self, domain_key: str):
        domain_name_full = config.DOMAINS[domain_key]

        id2str_map = self.id_map_full["item_dict"][domain_key]["id2str"]
        required_asins = set(id2str_map.values())
        meta_parquet_path = os.path.join(config.RAW_DATA_DIR, f"meta_{domain_name_full}.parquet")
        stream_amazon_meta_to_parquet(domain_name_full)
        meta_df = pd.read_parquet(meta_parquet_path)
        filtered_meta_df = meta_df[meta_df['asin'].isin(required_asins)]

        base_template = config.ITEM_PROMPT_TEMPLATE

        simple_domain_type = domain_name_full.lower()

        formatted_template = base_template.format(domain_type=simple_domain_type)

        item_prompts = self._build_prompts_from_parquet(filtered_meta_df,formatted_template=formatted_template)

        emb_cache_path = os.path.join(config.HANDLE_DATA_DIR, f"item_emb_cache_{domain_key}.pkl")
        item_embeddings = {}
        if os.path.exists(emb_cache_path):
            item_embeddings = pickle.load(open(emb_cache_path, "rb"))
        
        keys_to_process = [key for key in item_prompts.keys() if key not in item_embeddings]
        prompts_to_process = [item_prompts[key] for key in keys_to_process]

        if prompts_to_process:
            print(f"Processing {len(prompts_to_process)} items for domain {domain_key}")
            new_embeddings = self.embedder.get_embeddings(prompts_to_process)
            for key, emb in zip(keys_to_process, new_embeddings):
                item_embeddings[key] = emb.tolist()
            with open(emb_cache_path, "wb") as f: pickle.dump(item_embeddings, f)
        
        final_emb_list = []
        if item_embeddings:

            emb_dim = len(next(iter(item_embeddings.values())))

            for i in range(1, len(id2str_map) + 1):
                asin = id2str_map.get(str(i))
                embedding = item_embeddings.get(asin, [0] * emb_dim)
                final_emb_list.append(embedding)
            final_emb_np = np.array(final_emb_list)
            output_path = os.path.join(config.HANDLE_DATA_DIR, f"item_emb_{domain_key}.pkl")
            with open(output_path, "wb") as f: pickle.dump(final_emb_np, f)
            print(f"Saved item embeddings for domain {domain_key} to {output_path}")
        
        print(f" Building and saving title list for domain {domain_key}")

        asin_to_title = pd.Series(
            filtered_meta_df['title'].values,
            index = filtered_meta_df.asin
        ).to_dict()

        title_list = []

        for i in range(1, len(id2str_map) + 1):
            asin = id2str_map.get(str(i))
            title = asin_to_title.get(asin, "no_name")
            title_list.append(str(title)[:100])
        title_output_path = os.path.join(config.HANDLE_DATA_DIR, f"title_{domain_key}.pkl")
        with open(title_output_path ,"wb") as f:
            pickle.dump(title_list, f)
        print(f"Saved title list for domain {domain_key} to {title_output_path}")
    
