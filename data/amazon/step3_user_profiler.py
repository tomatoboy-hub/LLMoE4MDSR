import os
import pickle
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import copy
import shutil
from sklearn.cluster import KMeans
import config
from llm_services import LocalSummarizer, LocalEmbedder

class UserProfileDataset(Dataset):
    def __init__(self,partitioned_user_inter):
        self.user_ids = list(partitioned_user_inter.keys())
        self.partitioned_user_inter = partitioned_user_inter
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self,idx):
        user_id = self.user_ids[idx]
        partition_inter = self.partitioned_user_inter[user_id]
        return user_id, partition_inter

def user_profile_collate_fn(batch,title_list,prompt_template):
    batch_user_ids = []
    analyzer_prompts_all = []
    user_prompt_map = []

    for user_id, p_inter in batch:
        batch_user_ids.append(user_id)
        prompts_for_user = []
        for meta_inter in p_inter:
            if not meta_inter: continue
            temp_meta_inter = meta_inter[-15:]
            item_str = "\n".join([title_list[item_id - 1] for item_id in temp_meta_inter if item_id -1 < len(title_list)])
            prompts_for_user.append(prompt_template["analyzer"].format(item_str))
        analyzer_prompts_all.extend(prompts_for_user)
        user_prompt_map.append(len(prompts_for_user))

    return batch_user_ids, analyzer_prompts_all, user_prompt_map

class UserProfiler:
    def __init__(self, embedder: LocalEmbedder, summarizer: LocalSummarizer):
        self.embedder = embedder
        self.summarizer = summarizer
        self.n_clusters = config.NUM_CLUSTERS
        self.handled_dir = config.HANDLE_DATA_DIR
        
        # データを格納するインスタンス変数
        self.llm_emb = None
        self.inter_seq = None
        self.domain_seq = None
        self.item_num_dict = None
        self.title_list = None
        self.user_inter = {}
        self.cluster_map = {}
        self.partitioned_user_inter = {}
        self.user_profiles = {}
        self.user_embeddings = {}

    def _load_data(self):
        """ステップ1と2で生成したデータを読み込む""" 
        id_map = json.load(open(os.path.join(self.handled_dir, "id_map.json"), "r"))
        self.item_num_dict = id_map["item_dict"]["item_count"]
        

        all_embs = []
        all_titles = []
        for domain_key in sorted(config.DOMAINS.keys()):
            emb_path = os.path.join(self.handled_dir, f"item_emb_{domain_key}.pkl")
            title_path = os.path.join(self.handled_dir, f"title_{domain_key}.pkl")
            all_embs.append(pickle.load(open(emb_path,"rb")))
            all_titles.extend(pickle.load(open(title_path,"rb")))
        self.llm_emb = np.concatenate(all_embs)
        self.title_list = all_titles

        inter_path = os.path.join(self.handled_dir, config.INTER_FILE_NAME)
        self.inter_seq ,self.domain_seq = pickle.load(open(inter_path,"rb"))

        print("Data Loading complete")
    
    def _prepare_user_interactions(self):
        print("--- 3.2: Preparing user interaction data ---")
        offset, domain_offsets = 0,{}
        for key in sorted(self.item_num_dict.keys()):
            domain_offsets[key] = offset
            offset += self.item_num_dict[key]

        for user_str_id, seq in tqdm(self.inter_seq.items(), desc = "Preparing user interactions"):
            user = int(user_str_id)
            d_seq = self.domain_seq[user_str_id]

            meta_seq = np.array(copy.deepcopy(seq[:-1]))
            meta_domain = np.array(d_seq[:-1])

            unique_seq = [item_id + domain_offsets[str(domain_id)] for item_id, domain_id in zip(meta_seq, meta_domain)]
            self.user_inter[user] = unique_seq
        print("User interaction preparation complete")        
    
    def _process_items_with_resume(self,source_dict, save_filepath, processing_function, **kwargs):
        """進捗の読み込み、未処理アイテムのループ、処理、安全な保存を一般化した関数。"""
        temp_filepath = save_filepath + ".tmp"
        
        processed_results = {}
        if os.path.exists(save_filepath):
            try:
                if os.path.getsize(save_filepath) > 0:
                    with open(save_filepath, "rb") as f:
                        processed_results = pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                print(f"Warning: Cache file '{save_filepath}' is corrupt. It will be ignored.")
                processed_results = {}

        processed_keys = set(processed_results.keys())
        # source_dict のキーの型と processed_keys の型を合わせる
        keys_to_process = [key for key in source_dict.keys() if str(key) not in map(str, processed_keys)]

        print("-" * 50)
        print(f"Starting task for: {os.path.basename(save_filepath)}")
        print(f"Total items: {len(source_dict)}")
        print(f"Already processed: {len(processed_keys)}")
        print(f"Items to process: {len(keys_to_process)}")
        
        if not keys_to_process:
            print("All items have already been processed.")
            return processed_results

        try:
            with tqdm(total=len(keys_to_process), desc=f"Processing {os.path.basename(save_filepath)}") as pbar:
                for i, key in enumerate(keys_to_process):
                    item_value = source_dict[key]
                    result = processing_function(key, item_value, **kwargs)
                    if result is not None: # 処理が成功した場合のみ結果を保存
                        processed_results[key] = result
                    pbar.update(1)

                    if (i + 1) % 100 == 0:
                        with open(temp_filepath, "wb") as f:
                            pickle.dump(processed_results, f)
                        shutil.move(temp_filepath, save_filepath)
        finally:
            print(f"\nSaving final progress for {os.path.basename(save_filepath)}...")
            try:
                with open(temp_filepath, "wb") as f:
                    pickle.dump(processed_results, f)
                shutil.move(temp_filepath, save_filepath)
                print("Progress saved successfully.")
            except Exception as save_err:
                print(f"Failed to save final progress: {save_err}")

        return processed_results
        
    def _generate_user_profile(self, user_id, partition_inter):
        """
        1ユーザー分の分割済み行動履歴からLLMを使ってプロファイルを生成する。（バッチ処理対応版）
        """
        prompt_template = config.USER_PROFILE_PROMPTS
        
        # --- ステップ1: バッチ処理のために、まず全プロンプトをリストに集める --
        partition_pref = []
        for meta_inter in partition_inter:
            if not meta_inter:
                continue
            
            temp_meta_inter = meta_inter[-15:] if len(meta_inter) > 15 else meta_inter
            inter_str = "\n".join(self.title_list[item_id - 1] for item_id in temp_meta_inter)
            pref_prompt = prompt_template["analyzer"].format(inter_str)
            
            try:
                pref = self.summarizer.summarize(pref_prompt)
                if pref:
                    partition_pref.append(pref)
            except Exception as e:
                print(f"Error generating summary for user {user_id}. Error: {e}")
                return None
        
        if not partition_pref:
            return ""
        
        all_pref = "\n\n".join(partition_pref)
        summary_prompt = prompt_template["summarizer"].format(all_pref)
        try:
            summary = self.summarizer.summarize(summary_prompt)
            if summary:
                return summary
            else:
                return ""
        except Exception as e:
            print(f"Error generating summary for user {user_id}. Error: {e}")
            return ""



    def _generate_embedding(self, user_id, profile_text):
        if not profile_text:
            return None
        
        try:
            embedding = self.embedder.get_embeddings([profile_text])
            return embedding[0] if len(embedding) > 0 else None
        except Exception as e:
            print(f"Error generating embedding for user {user_id}. Error: {e}")
            return None
    def _cluster_items(self):
        print("Clustering items")
        model = KMeans(n_clusters=self.n_clusters, random_state=42)
        yhat = model.fit_predict(self.llm_emb)
        self.cluster_map = {
            item_id+1: cluster_id for item_id, cluster_id in enumerate(yhat, start=1)
        }
        print("Items clustered")
    
    def _partition_sequences(self):
        print("Partitioning sequences")
        for user, inter in tqdm(self.user_inter.items(), desc="Partition"):
            partition_inter = [[] for _ in range(self.n_clusters)]
            for item_id in inter:
                cluster_id = self.cluster_map.get(item_id)
                if cluster_id is not None:
                    partition_inter[cluster_id].append(item_id)
            self.partitioned_user_inter[user] = partition_inter
        print("User interactions partitioned")

    def _generate_profiles_with_dataloader(self):
        print("\n--- 3.5: Generating user profiles with DataLoader ---")
        user_profile_path = os.path.join(self.handled_dir, "user_profile.pkl")
        self.user_profiles = pickle.load(open(user_profile_path,"rb")) if os.path.exists(user_profile_path) else {}

        keys_to_process = {uid:p_inter for uid, p_inter in self.partitioned_user_inter.items() if uid not in self.user_profiles}

        if not keys_to_process:
            print("All user profile have been processed")
            return 

        print(f"Total users: {len(self.partitioned_user_inter)}, Already processed: {len(self.user_profiles)}, To process: {len(keys_to_process)}")
        
        dataset = UserProfileDataset(keys_to_process)
        collate_fn_with_args = lambda batch: user_profile_collate_fn(batch, self.title_list, config.USER_PROFILE_PROMPTS)

        data_loader = DataLoader(dataset, batch_size=config.PROFILE_BATCH_SIZE, shuffle=False, num_workers = 4, collate_fn = collate_fn_with_args)

        for batch_user_ids, analyser_prompts_all, user_prompt_map in tqdm(data_loader, desc="Processing User Profile Batches"):

            generated_prefs = self.summarizer.summarize_batch(analyser_prompts_all, batch_size=config.LLM_BATCH_SIZE)
            # ステージ2: 結果を統合し、最終要約プロンプトを作成 (ここはCPU処理)
            summarizer_prompts_final = []
            pref_idx = 0
            for pref_count in user_prompt_map:
                if pref_count > 0:
                    user_prefs = generated_prefs[pref_idx : pref_idx + pref_count]
                    all_pref_str = "\n\n".join(filter(None, user_prefs))
                    summarizer_prompts_final.append(config.USER_PROFILE_PROMPTS["summarizer"].format(all_pref_str))
                    pref_idx += pref_count
                else:
                    summarizer_prompts_final.append("")
            # ステージ3: 全ての最終要約プロンプトを一度にGPUに送る
            final_summaries = self.summarizer.summarize_batch(summarizer_prompts_final, batch_size=config.LLM_BATCH_SIZE)
            # 結果を保存
            for user_id, summary in zip(batch_user_ids, final_summaries):
                if summary: self.user_profiles[user_id] = summary
            #最終保存
        with open(user_profile_path, "wb") as f: pickle.dump(self.user_profiles, f)
        print("User profile generation complete.")
    
    def _generate_embeddings_in_batches(self):
        """生成されたプロファイルの埋め込みを効率的に作成する。"""
        print("\n--- 3.6: Generating embeddings for user profiles ---")
        user_emb_path = os.path.join(self.handled_dir, "user_profile_emb.pkl")
        self.user_embeddings = pickle.load(open(user_emb_path, "rb")) if os.path.exists(user_emb_path) else {}
        
        keys_for_embedding = [uid for uid in self.user_profiles if uid not in self.user_embeddings]

        if not keys_for_embedding:
            print("  All embeddings have been generated.")
            return

        print(f"  Users to process for embeddings: {len(keys_for_embedding)}")
        
        profile_texts = [self.user_profiles[uid] for uid in keys_for_embedding]
        new_embeddings = self.embedder.get_embeddings(profile_texts, batch_size=config.PROFILE_BATCH_SIZE * 2)
        
        for user_id, emb in zip(keys_for_embedding, new_embeddings):
            self.user_embeddings[user_id] = emb

        with open(user_emb_path, "wb") as f: pickle.dump(self.user_embeddings, f)
        print("  User profile embedding complete.")
    
    def _save_final_embeddings(self):
        """最終的なNumpy配列を保存する。"""
        print("\n--- 3.7: Saving final user embedding array ---")
        # ユーザーIDの順序を保つためにソート
        all_user_ids = sorted(self.partitioned_user_inter.keys())
        
        final_emb_list = [self.user_embeddings[uid] for uid in all_user_ids if uid in self.user_embeddings]
        
        if not final_emb_list:
            print("  No embeddings to save.")
            return
            
        final_emb_array = np.array(final_emb_list)
        final_emb_path = os.path.join(self.handled_dir, "usr_profile_emb_final.pkl")
        with open(final_emb_path, "wb") as f: pickle.dump(final_emb_array, f)
        print(f"  Saved final {len(final_emb_array)} user profile embeddings to {final_emb_path}")
        
    def run_pipeline(self):
        self._load_data()
        self._prepare_user_interactions()
        self._cluster_items()
        self._partition_sequences()
        print("\n--- 3.5: Saving partitioned user sequences for inspection ---")
        partitioned_path = os.path.join(self.handled_dir, "partitioned_user_sequences.pkl")
        with open(partitioned_path, "wb") as f:
            pickle.dump(self.partitioned_user_inter, f)
        print(f"  Saved partitioned data to {partitioned_path}")
        # 👈【修正点】古い処理を削除し、新しいDataLoaderベースの処理を呼び出す
        self._generate_profiles_with_dataloader()
        self._generate_embeddings_in_batches()
        self._save_final_embeddings()