import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
import config
from tqdm import tqdm


class DimensionalityReducer:
    """
    アイテムの埋め込みベクトルをPCAで次元削減するクラス。
    ドメインごとの削減と、全ドメインを結合した削減の両方を行う。
    """
    def __init__(self):
        self.handled_dir = config.HANDLE_DATA_DIR
        self.domains = config.DOMAINS
        self.pca_dimension = config.PCA_TARGET_DIMENSION
        
        # 読み込んだ埋め込みを格納する辞書
        self.embeddings = {}

    def run_pipeline(self):
        """次元削減の全ステップを順番に実行するメインメソッド。"""
        self._load_embeddings()
        self._reduce_separate()
        self._reduce_combined()
        print("  Dimensionality reduction pipeline finished successfully.")

    def _load_embeddings(self):
        """ステップ2で生成された、ドメインごとの埋め込みファイルを読み込む。"""
        print("  Loading item embeddings for all domains...")
        for domain_key in self.domains.keys():
            emb_path = os.path.join(self.handled_dir, f"item_emb_{domain_key}.pkl")
            if os.path.exists(emb_path):
                with open(emb_path, "rb") as f:
                    self.embeddings[domain_key] = pickle.load(f)
            else:
                print(f"    Warning: Embedding file not found for domain {domain_key} at {emb_path}. Skipping.")
        print(f"  Loaded embeddings for {len(self.embeddings)} domains.")

    def _reduce_separate(self):
        """ドメインごとに個別にPCAを適用し、結果を保存する。"""
        print(f"  Reducing dimensions for each domain separately to {self.pca_dimension}...")
        for domain_key, emb_array in tqdm(self.embeddings.items(), desc="  Processing separate PCA"):
            if not emb_array.any(): continue
            n_samples = emb_array.shape[0]
            
            # 👈【ここからが重要修正箇所】
            if n_samples < self.pca_dimension:
                print(f"\n    [Warning] Domain {domain_key} has only {n_samples} items, which is less than the target PCA dimension {self.pca_dimension}.")
                
                # 1. まず、可能な最大次元数 (n_samples) でPCAを実行
                pca = PCA(n_components=n_samples)
                pca_emb_small = pca.fit_transform(emb_array) # 出力形状: (n_samples, n_samples)
                
                # 2. 目標次元数のゼロ行列を作成
                pca_emb = np.zeros((n_samples, self.pca_dimension))
                
                # 3. ゼロ行列に、PCAの結果を左詰めでコピー（ゼロパディング）
                pca_emb[:, :n_samples] = pca_emb_small
                print(f"    Padded embeddings from shape {pca_emb_small.shape} to {pca_emb.shape}.")
                
            else:
                # サンプル数が十分な場合は、通常通りPCAを実行
                pca = PCA(n_components=self.pca_dimension)
                pca_emb = pca.fit_transform(emb_array)
            # 結果を保存
            output_filename = f"itm_emb_np_{domain_key}_pca{self.pca_dimension}.pkl"
            output_path = os.path.join(self.handled_dir, output_filename)
            with open(output_path, "wb") as f:
                pickle.dump(pca_emb, f)
        print("  Separate PCA complete.")

    def _reduce_combined(self):
        """全ドメインの埋め込みを結合し、単一のPCAを適用して結果を保存する。"""
        print(f"  Reducing dimensions for all combined domains to {self.pca_dimension}...")
        if not self.embeddings:
            print("    No embeddings to combine. Skipping.")
            return

        # 全ての埋め込み配列を結合
        all_emb = np.concatenate(list(self.embeddings.values()), axis=0)
        print(f"    Combined array shape: {all_emb.shape}")
        
        # 単一のPCAモデルで学習と変換を行う
        pca = PCA(n_components=self.pca_dimension)
        pca_emb_all = pca.fit_transform(all_emb)
        
        # 結果を保存
        output_filename = f"itm_emb_np_all.pkl"
        output_path = os.path.join(self.handled_dir, output_filename)
        with open(output_path, "wb") as f:
            pickle.dump(pca_emb_all, f)
        print(f"  Combined PCA complete. Result saved to {output_path}")
