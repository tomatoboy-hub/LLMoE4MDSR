import os
import pickle
import numpy as np
import json
from tqdm import tqdm
import copy
import shutil
from sklearn.cluster import KMeans
import config
from llm_services import LocalSummarizer, LocalEmbedder

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
        
        for user, inter in tqdm(self.user_inter.items(), desc="Partitioning sequences"):
            partition_inter = [[] for _ in range(self.n_clusters)]
            for item_id in inter:
                cluster_id = self.cluster_map.get(item_id)
                if cluster_id is not None:
                    partition_inter[cluster_id].append(item_id)
            self.partitioned_user_inter[user] = partition_inter
        print("User interactions partitioned")
    
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
        """1ユーザー分の分割済み行動履歴からLLMを使ってプロファイルを生成する。"""
    
        prompt_template = config.USER_PROFILE_PROMPTS

        
        partition_pref = []
        # 各クラスタの行動履歴から嗜好を分析
        for meta_inter in partition_inter:
            if not meta_inter:
                continue
            
            # アイテム数が多すぎる場合は最新15件に絞る
            temp_meta_inter = meta_inter[-15:] if len(meta_inter) > 15 else meta_inter
            
            # IDを商品タイトルに変換
            inter_str = "\n".join(self.title_list[item_id - 1] for item_id in temp_meta_inter)
            
            pref_prompt = prompt_template["analyzer"].format(inter_str)
            try:
                pref = self.summarizer.summarize(pref_prompt)
                if pref:
                    partition_pref.append(pref)
            except Exception as e:
                print(f"Error generating preference for user {user_id}, cluster part. Error: {e}")

        if not partition_pref:
            return "" # 分析できる嗜好がなければ空文字を返す

        # 全クラスタの嗜好を統合して最終的なプロファイルを生成
        all_pref = "\n\n".join(partition_pref)
        summary_prompt = prompt_template["summarizer"].format(all_pref)
        try:
            summary = self.summarizer(summary_prompt)
            return summary
        except Exception as e:
            print(f"Error generating summary for user {user_id}. Error: {e}")
            return "" # 統合に失敗した場合は空文字を返す
        
    def _generate_embedding(self, user_id, profile_text):
        if not profile_text:
            return None
        
        try:
            embedding = self.embedder.get_embedding([profile_text])
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

        
    def run_pipeline(self):
        self._load_data()
        self._prepare_user_interactions()
        self._cluster_items()
        self._partition_sequences()

        user_profile_path = os.path.join(self.handled_dir, "user_profile.pkl")
        self.user_profiles = self._process_items_with_resume(
            source_dict = self.partitioned_user_inter,
            save_filepath = user_profile_path,
            processing_function = self._generate_user_profile)
        
        user_emb_path = os.path.join(self.handled_dir, "user_profile_emb.pkl")
        self.user_embeddings = self._process_items_with_resume(
            source_dict = self.user_profiles,
            save_filepath = user_emb_path,
            processing_function = self._generate_embedding)

        emb_list = []
        for user_id in sorted(self.user_embeddings.keys()):
            embedding = self.user_embeddings[user_id]
            if embedding is not None and len(embedding) > 0:
                emb_list.append(embedding)
            
        final_emb_array = np.array(emb_list)
        final_emb_path = os.path.join(self.handled_dir, "user_profile_emb_final.pkl")
        with open(final_emb_path, "wb") as f:
            pickle.dump(final_emb_array, f)
        print(f"Saved final user profile embeddings to {final_emb_path}")
        print("Pipeline completed successfully")
