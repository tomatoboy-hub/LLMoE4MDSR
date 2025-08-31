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
DATASET_NAME = "elec"
INTER_FILE = f"{DATASET_NAME}_phone"
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
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - {response.text}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

def get_profile_summary(prompt_text):
    """GPT-3.5を呼び出して、プロファイルの要約テキストを取得する。"""
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.5,
        "max_tokens": 1024,
    }
    re_json = call_openai_api(url, payload)
    if "choices" in re_json and re_json["choices"]:
        return re_json["choices"][0]["message"]["content"]
    else:
        raise ValueError(f"Invalid API response for profile summary: {re_json}")

def get_embedding_vector(text):
    """Embeddingモデルを呼び出して、テキストの埋め込みベクトルを取得する。"""
    url = "https://api.openai.com/v1/embeddings"
    payload = {"model": "text-embedding-ada-002", "input": text}
    re_json = call_openai_api(url, payload)
    if "data" in re_json and re_json["data"]:
        return re_json["data"][0]["embedding"]
    else:
        raise ValueError(f"Invalid API response for embedding: {re_json}")

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
    keys_to_process = [key for key in source_dict.keys() if key not in processed_keys]

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

# --- 5. メインの処理ロジック ---

def main():
    """スクリプト全体の処理を実行するメイン関数。"""
    
    # --- 5.1 データ読み込み ---
    print("--- Loading data ---")
    domainA_emb_path = os.path.join(DATA_DIR, f"{EMB_FILE_PREFIX}_A.pkl")
    domainB_emb_path = os.path.join(DATA_DIR, f"{EMB_FILE_PREFIX}_B.pkl")
    inter_seq_path = os.path.join(DATA_DIR, f"{INTER_FILE}.pkl")
    id_map_path = os.path.join(DATA_DIR, "id_map.json")
    title_A_path = os.path.join(DATA_DIR, "title_A.pkl")
    title_B_path = os.path.join(DATA_DIR, "title_B.pkl")

    llm_emb_A = pickle.load(open(domainA_emb_path, "rb"))
    llm_emb_B = pickle.load(open(domainB_emb_path, "rb"))
    llm_emb = np.concatenate([llm_emb_A, llm_emb_B])

    inter_seq, domain_seq = pickle.load(open(inter_seq_path, 'rb'))
    id_map = json.load(open(id_map_path, "r"))
    item_num_dict = id_map["item_dict"]["item_count"]
    
    title_A = pickle.load(open(title_A_path, "rb"))
    title_B = pickle.load(open(title_B_path, "rb"))
    title_list = title_A + title_B

    # --- 5.2 ユーザーの行動履歴を準備 ---
    print("--- Preparing user interaction data ---")
    user_inter = {}
    for user in tqdm(inter_seq.keys(), desc="Preparing user interactions"):
        meta_seq = np.array(copy.deepcopy(inter_seq[user][:-1]))
        meta_domain = np.array(domain_seq[user][:-1])
        # ドメインBのアイテムIDにドメインAのアイテム数を加算してIDをユニークにする
        meta_seq[meta_domain == 1] += item_num_dict["0"]
        user_inter[user] = meta_seq.tolist()

    # --- 5.3 K-Meansクラスタリングでアイテムをグループ化 ---
    print("--- Clustering items with K-Means ---")
    model = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    yhat = model.fit_predict(llm_emb)
    # ▼▼▼【ここから確認コードを追加】▼▼▼
    
    # 方法1：NumPyを使う方法（推奨）
    unique_clusters, counts = np.unique(yhat, return_counts=True)
    
    print("\n--- Cluster Size Verification ---")
    print("各クラスタに属しているアイテムの数:")
    for cluster_id, count in zip(unique_clusters, counts):
        print(f"  Cluster {cluster_id}: {count} items")
    print("---------------------------------\n")

    from sklearn.manifold import TSNE
    from collections import Counter
    import re
    import matplotlib.pyplot as plt

    # ▼▼▼【5.3のブロックの最後、cluster_mapを定義した後に、このコードブロックを追加】▼▼▼

    # --- 5.3.1 クラスタの可視化と解釈 ---
    print("\n--- Starting Cluster Visualization and Interpretation ---")

    # 1. t-SNEによる次元削減 (高次元のEmbeddingを2次元に圧縮)
    print("Running t-SNE for dimensionality reduction... (This may take a few minutes)")
    tsne = TSNE(n_components=2, random_state=1)
    emb_2d = tsne.fit_transform(llm_emb)
    print("t-SNE finished.")

    # 2. 2D散布図による可視化
    print("Generating cluster scatter plot...")
    plt.figure(figsize=(12, 10))
    # クラスタごとに色を分けてプロット
    for cluster_id in range(N_CLUSTERS):
        # 現在のクラスタに属するアイテムのインデックスを取得
        indices = np.where(yhat == cluster_id)[0]
        # 対応する2D座標をプロット
        plt.scatter(emb_2d[indices, 0], emb_2d[indices, 1], label=f'Cluster {cluster_id}', alpha=0.6)

    plt.title('t-SNE Visualization of Item Clusters')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True)
    # 画像ファイルとして保存
    visualization_path = os.path.join(DATA_DIR, 'cluster_visualization.png')
    plt.savefig(visualization_path)
    print(f"Cluster visualization saved to {visualization_path}")
    # plt.show() # Jupyter Notebookなどで直接表示したい場合

    # 3. 各クラスタのキーワード抽出による解釈
    print("\n--- Extracting Keywords for Each Cluster ---")

    # 簡単な英語のストップワードリスト (よく出るけど意味のない単語を除外)
    stop_words = set([
        'a', 'an', 'and', 'the', 'in', 'on', 'for', 'with', 'to', 'of', 'is', 'it', 'you', 
        'i', 'he', 'she', 'they', 'we', 'are', 'was', 'were', 'be', 'been', 'has', 'have', 
        'had', 'do', 'does', 'did', 'but', 'if', 'or', 'as', 'by', 'from', 'at', 'this', 
        'that', 'these', 'those', 'not', 'no', 'new', 'old', 'pro', 'for', 'pack'
    ])

    for cluster_id in range(N_CLUSTERS):
        # 現在のクラスタに属するアイテムのインデックスを取得
        indices = np.where(yhat == cluster_id)[0]
        
        # 対応する商品タイトルをすべて集める
        cluster_titles = [title_list[i] for i in indices]
        
        # 全タイトルを一つのテキストに結合し、単語に分割
        all_words = []
        for title in cluster_titles:
            # 数字や記号を削除し、小文字に変換
            words = re.findall(r'\b\w+\b', title.lower())
            all_words.extend([word for word in words if word not in stop_words and not word.isdigit()])
        
        # 最も頻繁に出現する単語をカウント
        word_counts = Counter(all_words)
        most_common_words = word_counts.most_common(10) # 上位10個のキーワード
        
        print(f"\n[Cluster {cluster_id}]")
        print(f"  Total Items: {len(indices)}")
        print(f"  Top Keywords: {[word for word, count in most_common_words]}")

    print("\n--- Cluster interpretation finished ---")

    # 方法2：collections.Counterを使う方法（別解）
    # from collections import Counter
    # cluster_counts = Counter(yhat)
    # print("\n--- Cluster Size Verification (Counter) ---")
    # print(cluster_counts)
    # print("---------------------------------\n")

    # ▲▲▲【確認コードはここまで】▲▲▲
    
#     cluster_map = {item_id + 1: cluster_id for item_id, cluster_id in enumerate(yhat)}
    
#     # --- 5.4 ユーザーの行動履歴をクラスタごとに分割 ---
#     print("--- Partitioning user interactions by cluster ---")
#     partitioned_user_inter = {}
#     for user, inter in user_inter.items():
#         partition_inter = [[] for _ in range(N_CLUSTERS)]
#         for item_id in inter:
#             partition_inter[cluster_map[item_id]].append(item_id)
#         partitioned_user_inter[user] = partition_inter

#     # --- 5.5 ユーザープロファイルの生成（2ステップに分割） ---

#     prompt_template = {
#         "summarizer": "Assume you are an consumer and there are preference demonstrations from several aspects are as follows:\n {}. \n\n Please illustrate your preference with less than 100 words",
#         "analyzer": "Assume you are a consumer who is shopping online. You have shown interests in following commdities:\n {}. \n\n The commodities are segmented by '\n'. \n\n Please conclude it not beyond 50 words. Do not only evaluate one specfic commodity but illustrate the interests overall."
#     }

#     # --- ステップ5.5.1：【1段階目】カテゴリごとの興味の要約を生成 ---

#     def generate_partition_preferences(user_id, partition_inter, title_list, prompt_template):
#         """
#         【1段階目の関数】
#         一人のユーザーの分割された行動履歴から、「カテゴリごとの興味の要約リスト」を生成する。
#         """
#         partition_pref_list = []
#         for meta_inter in partition_inter:
#             if not meta_inter: continue
#             temp_meta_inter = meta_inter[-15:]
#             inter_str = " \n".join([title_list[item_id - 1] for item_id in temp_meta_inter])
            
#             pref_prompt = prompt_template["analyzer"].format(inter_str)
#             pref = get_profile_summary(pref_prompt) # AnalyzerのLLM呼び出し
#             partition_pref_list.append(pref)
        
#         return partition_pref_list
    
#     # ▼▼▼【テスト実行のためのコード】▼▼▼
#     # このブロックを追加すると、最初の5人のユーザーだけでテストできます。
#     # 全員で実行したい場合は、このブロックをコメントアウト（各行の先頭に#を付ける）か削除してください。
#     NUM_TEST_USERS = 5 
#     partitioned_user_inter = {
#         k: partitioned_user_inter[k] for k in list(partitioned_user_inter.keys())[:NUM_TEST_USERS]
#     }
#     print(f"\n--- !!! TEST MODE: Processing only the first {NUM_TEST_USERS} users. !!! ---\n")
#     # ▲▲▲【テスト実行のためのコードはここまで】▲▲▲
#     # process_items_with_resume を使って、1段階目の処理を実行
#     # 新しいファイル "partition_preferences.pkl" に中間結果を保存
#     partition_preferences = process_items_with_resume(
#         source_dict=partitioned_user_inter,
#         save_filepath=os.path.join(DATA_DIR, "partition_preferences.pkl"),
#         processing_function=generate_partition_preferences,
#         # processing_function に渡す追加の引数を指定
#         title_list=title_list,
#         prompt_template=prompt_template
#     )


# # --- ステップ5.5.2：【2段階目】最終的なユーザープロファイルの生成 ---

#     def summarize_final_profile(user_id, partition_pref_list, prompt_template):
#         """
#         【2段階目の関数】
#         「カテゴリごとの興味の要約リスト」から、最終的なプロファイルを生成する。
#         """
#         if not partition_pref_list:
#             return "No specific preference found."

#         all_pref = " \n".join(partition_pref_list)
#         summary_prompt = prompt_template["summarizer"].format(all_pref)
#         summary = get_profile_summary(summary_prompt) # SummarizerのLLM呼び出し
#         return summary

#     # 再び process_items_with_resume を使用。今度は1段階目の結果を入力(source_dict)とする
#     user_profiles = process_items_with_resume(
#         source_dict=partition_preferences,
#         save_filepath=os.path.join(DATA_DIR, "user_profile.pkl"),
#         processing_function=summarize_final_profile,
#         # processing_function に渡す追加の引数を指定
#         prompt_template=prompt_template
#     )
    

#     # --- 5.6 ユーザープロファイルのEmbedding生成 ---
#     def generate_user_embedding(user_id, profile_text):
#         """一人のユーザーのプロファイルテキストから、Embeddingを生成する関数"""
#         return get_embedding_vector(profile_text)

#     user_embeddings = process_items_with_resume(
#         source_dict=user_profiles,
#         save_filepath=os.path.join(DATA_DIR, "usr_profile_emb.pkl"),
#         processing_function=generate_user_embedding
#     )

#     # --- 5.7 最終的なNumpy配列の作成と保存 ---
#     print("--- Creating and saving final user embedding array ---")
#     final_emb_list = []
#     sorted_user_ids = sorted(user_embeddings.keys())
#     for user_id in sorted_user_ids:
#         final_emb_list.append(user_embeddings[user_id])
#     final_emb_array = np.array(final_emb_list)

#     final_save_path = os.path.join(DATA_DIR, "user_emb.pkl")
#     with open(final_save_path, "wb") as f:
#         pickle.dump(final_emb_array, f)
    
#     print(f"\nAll processing finished. Final user embeddings saved to {final_save_path}")
#     assert len(final_emb_array) == len(inter_seq)

# --- 6. スクリプトの実行 ---
if __name__ == '__main__':
    main()