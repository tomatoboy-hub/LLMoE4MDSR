import os 
import json
import pickle
import numpy as np 
from tqdm import tqdm
import copy 
import config
from llm_services import LocalEmbedder

class ItemFeatureExtractor:
    def __init__(self, embedder: LocalEmbedder):
        self.embedder = embedder
        self.id_map_full = json.load(open(os.path.join(config.HANDLE_DATA_DIR, "id_map.json"), "r"))

    def _build_prompts(self, raw_meta:dict):
        item_prompts = []
        for key,value in tqdm(raw_meta.items(),desc="Building prompts"):
            item_str = copy.deepcopy(config.ITEM_PROMPT_TEMPLATE)
            for attr in ["title", "brand", "date","price"]:
                attr_val = value.get(attr,"unknown")
                item_str = item.str.replace(f"<{attr.upper()}>",str(attr_val)[:100])
            item_prompts[key] = item_str
        return item_prompts
    
    def run_for_domain(self, domain_key: str):
        domain_name_full = config.DOMAINS[domain_key]

        raw_meta_path = os.path.join(config.HANDLED_DATA_DIR, f"item2attributes_{domain_name_full}.json")
        if not os.path.exists(raw_meta_path):
            print(f"Error: {raw_meta_path} does not exist")
            return 
        
        raw_meta = json.load(open(raw_meta_path, "r"))
        id2str_map = self.id_map_full["item_dict"][domain_key]["id2str"]

        item_prompts = self._build_prompts(raw_meta)
        emb_cache_path = os.path.join(config.HANDLED_DATA_DIR, f"item_emb_{domain_key}.pkl")
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
        emb_dim = len(next(iter(item_embeddings.values())))

        for i in range(1, len(id2str_map) + 1):
            asin = id2str_map.get(str(i))
            embedding = item_embeddings.get(asin, [0] * emb_dim)
            final_emb_list.append(embedding)
        final_emb_np = np.array(final_emb_list)
        output_path = os.path.join(config.HANDLED_DATA_DIR, f"item_emb_{domain_key}.pkl")
        with open(output_path, "wb") as f: pickle.dump(final_emb_np, f)
        print(f"Saved item embeddings for domain {domain_key} to {output_path}")
    
