# -*- coding: utf-8 -*-
"""
ユーザーの行動履歴からクラスタリングに基づいたプロファイルを生成し、
さらにそのプロファイルのEmbedding（ベクトル表現）を生成するスクリプト。

これまでの議論に基づき、以下の改善が施されています:
1. API呼び出しの堅牢化: tenacityによる自動リトライ機能
2. 進捗保存の安全性向上: 一時ファイルを利用したアトミックな保存
3. 処理の効率化: 再開時に未処理の項目のみをループ
4. コードの再利用性と可読性向上: 処理の関数化
"""

# --- 1. ライブラリのインポート ---
import os
import pickle
import numpy as np
import json
from tqdm import tqdm
import copy
import requests
import shutil
from tenacity import retry, wait_random_exponential, stop_after_attempt
from sklearn.cluster import KMeans

# --- 2. 設定とAPIキー ---


OPENAI_API_KEY = "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"

# データセットに関する設定
DATA_DIR = "./handled/"
DATASET_NAME = "cloth"
INTER_FILE = f"{DATASET_NAME}_sport_fashion"
EMB_FILE_PREFIX = "itm_emb_np"
N_CLUSTERS = 10  # クラスタ数
# --- 3. 堅牢なAPI呼び出し共通関数 ---

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_openai_api(url, payload):
    """OpenAI APIを自動リトライ機能付きで呼び出す、共通の堅牢な関数。"""
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # 4xx or 5xx エラーの場合は例外を発生
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An unexpected API error occurred: {e}")
        raise

def get_profile_summary(prompt_text):
    """GPT-3.5を呼び出して、プロファイルの要約テキストを取得する。"""
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.5,
        "max_tokens": 150, # 要約なのでトークン数を制限
    }
    re_json = call_openai_api(url, payload)
    if "choices" in re_json and re_json["choices"]:
        return re_json["choices"][0]["message"]["content"]
    else:
        print(f"Warning: Invalid API response for profile summary: {re_json}")
        return "" # 空の文字列を返して処理を続行

def get_embedding_vector(text):
    """Embeddingモデルを呼び出して、テキストの埋め込みベクトルを取得する。"""
    if not text: # 入力テキストが空の場合はAPIを呼ばない
        return []
    url = "https://api.openai.com/v1/embeddings"
    payload = {"model": "text-embedding-ada-002", "input": text}
    re_json = call_openai_api(url, payload)
    if "data" in re_json and re_json["data"]:
        return re_json["data"][0]["embedding"]
    else:
        print(f"Warning: Invalid API response for embedding: {re_json}")
        return [] # 空のリストを返して処理を続行

# --- 4. 安全なループ処理と保存を行う共通関数 ---

def process_items_with_resume(source_dict, save_filepath, processing_function, **kwargs):
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

# --- 5. LLM処理のための個別関数 ---

def generate_user_profile(user_id, partition_inter, title_list):
    """1ユーザー分の分割済み行動履歴からLLMを使ってプロファイルを生成する。"""
    
    prompt_template = {
        "analyzer": "Assume you are a consumer who is shopping online. You have shown interests in following commodities:\n{}\n\nThe commodities are segmented by '\\n'.\n\nPlease conclude it not beyond 50 words. Do not only evaluate one specific commodity but illustrate the interests overall.",
        "summarizer": "Assume you are a consumer and there are preference demonstrations from several aspects are as follows:\n{}\n\nPlease illustrate your final integrated preference with less than 100 words."
    }
    
    partition_pref = []
    # 各クラスタの行動履歴から嗜好を分析
    for meta_inter in partition_inter:
        if not meta_inter:
            continue
        
        # アイテム数が多すぎる場合は最新15件に絞る
        temp_meta_inter = meta_inter[-15:] if len(meta_inter) > 15 else meta_inter
        
        # IDを商品タイトルに変換
        inter_str = "\n".join([title_list[item_id - 1] for item_id in temp_meta_inter])
        
        pref_prompt = prompt_template["analyzer"].format(inter_str)
        try:
            pref = get_profile_summary(pref_prompt)
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
        summary = get_profile_summary(summary_prompt)
        return summary
    except Exception as e:
        print(f"Error generating summary for user {user_id}. Error: {e}")
        return "" # 統合に失敗した場合は空文字を返す

def generate_embedding(user_id, profile_text):
    """1ユーザーのプロファイルテキストからEmbeddingを生成する。"""
    try:
        return get_embedding_vector(profile_text)
    except Exception as e:
        print(f"Error generating embedding for user {user_id}. Error: {e}")
        return None


# --- 5. メインの処理ロジック ---

def main():
    """スクリプト全体の処理を実行するメイン関数。"""
    
    # --- 5.1 データ読み込み ---
    print("--- Loading data ---")
    domainA_emb_path = os.path.join(DATA_DIR, f"{EMB_FILE_PREFIX}_A.pkl")
    domainB_emb_path = os.path.join(DATA_DIR, f"{EMB_FILE_PREFIX}_B.pkl")
    domeinC_emb_path = os.path.join(DATA_DIR, f"{EMB_FILE_PREFIX}_C.pkl")
    inter_seq_path = os.path.join(DATA_DIR, f"{INTER_FILE}.pkl")
    id_map_path = os.path.join(DATA_DIR, "id_map.json")
    title_A_path = os.path.join(DATA_DIR, "title_A.pkl")
    title_B_path = os.path.join(DATA_DIR, "title_B.pkl")
    title_C_path = os.path.join(DATA_DIR, "title_C.pkl")

    llm_emb_A = pickle.load(open(domainA_emb_path, "rb"))
    llm_emb_B = pickle.load(open(domainB_emb_path, "rb"))
    llm_emb_C = pickle.load(open(domeinC_emb_path, "rb"))
    llm_emb = np.concatenate([llm_emb_A, llm_emb_B, llm_emb_C])

    inter_seq, domain_seq = pickle.load(open(inter_seq_path, 'rb'))
    id_map = json.load(open(id_map_path, "r"))
    item_num_dict = id_map["item_dict"]["item_count"]
    
    title_A = pickle.load(open(title_A_path, "rb"))
    title_B = pickle.load(open(title_B_path, "rb"))
    title_C = pickle.load(open(title_C_path, "rb"))

    title_list = title_A + title_B + title_C
    # --- 5.2 ユーザーの行動履歴を準備 ---
    print("--- Preparing user interaction data ---")
    user_inter = {}
    for user in tqdm(inter_seq.keys(), desc="Preparing user interactions"):
        meta_seq = np.array(copy.deepcopy(inter_seq[user][:-1]))
        meta_domain = np.array(domain_seq[user][:-1])
        # ドメインBのアイテムIDにドメインAのアイテム数を加算してIDをユニークにする
        meta_seq[meta_domain == 1] += item_num_dict["0"]
        meta_seq[meta_domain == 2] += item_num_dict["0"] + item_num_dict["1"]
        user_inter[user] = meta_seq.tolist()

    # --- 5.3 K-Meansクラスタリングでアイテムをグループ化 ---
    print("--- Clustering items with K-Means ---")
    model = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    yhat = model.fit_predict(llm_emb)

    #アイテムIDからクラスタIDへのマッピングを作成
    cluster_map = {item_id + 1: cluster_id for item_id, cluster_id in enumerate(yhat)}
    print("Item clustering complete.")

    # --- 6.4 行動履歴をクラスタごとに分割 ---
    print("\n--- 4. Partitioning user interactions by cluster ---")
    partitioned_user_inter = {}
    for user, inter in tqdm(user_inter.items(), desc="Partitioning sequences"):
        partition_inter = [[] for _ in range(N_CLUSTERS)]
        for item_id in inter:
            cluster_id = cluster_map.get(item_id)
            if cluster_id is not None:
                partition_inter[cluster_id].append(item_id)
        partitioned_user_inter[user] = partition_inter
    print("User interactions partitioned.")

    print("--- Saving partitioned user interactions ---")
    user_profile_path = os.path.join(DATA_DIR, f"user_profile.pkl")
    user_profiles = process_items_with_resume(
        source_dict=partitioned_user_inter,
        save_filepath=user_profile_path,
        processing_function=generate_user_profile,
        title_list=title_list
    )
    print("User profile generation complete")

    print("Generating embeddings for users profile")
    user_emb_path = os.path.join(DATA_DIR, f"user_profile_emb.pkl")
    user_emb_dict = process_items_with_resume(
        source_dict=user_profiles,
        save_filepath=user_emb_path,
        processing_function=generate_embedding
    )
    emb_list = []
    # 元のユーザー順序を維持するために sorted を使用
    for user_id in sorted(user_embeddings_dict.keys()):
        embedding = user_embeddings_dict[user_id]
        if embedding: # Embeddingが正常に生成されたものだけを追加
            emb_list.append(embedding)

    final_emb_array = np.array(emb_list)
    final_emb_path = os.path.join(DATA_DIR, "usr_profile_emb_final.pkl")
    with open(final_emb_path, "wb") as f:
        pickle.dump(final_emb_array, f)
    
    print(f"\n--- All processes complete! ---")
    print(f"Total {len(final_emb_array)} user profile embeddings saved to {final_emb_path}")


# --- 6. スクリプトの実行 ---
if __name__ == '__main__':
    main()