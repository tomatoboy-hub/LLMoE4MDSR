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
            emb_path = os.path.join(self.handled_dir, f"itm_emb_np_{domain_key}.pkl")
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
            # PCAモデルを初期化し、学習と変換を同時に行う
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
