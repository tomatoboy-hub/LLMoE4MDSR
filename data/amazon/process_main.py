import os 
import config

from llm_services import LocalSummarizer, LocalEmbedder
from step1_data_processor import AmazonHandler
from step2_feature_extractor import ItemFeatureExtractor
from step3_user_profiler import UserProfiler
from step4_pca_reducer import DimensionalityReducer
from huggingface_hub import login

# --- 認証処理 ---
# 1. 環境変数 HUGGING_FACE_HUB_TOKEN からトークンを読み込むことを試みる
# 2. もし環境変数が設定されていなければ、以下の文字列 "hf_..." を直接使う
#    (セキュリティのため、可能な限り環境変数の利用を推奨します)
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or "hf_YOUR_ACCESS_TOKEN_HERE"


def main():
    print("=" * 50)
    print("Starting data processing pipeline")
    print("=" * 50)

    step1_output_path = os.path.join(config.HANDLE_DATA_DIR, config.INTER_FILE_NAME)
    if not os.path.exists(step1_output_path):
        data_processor = AmazonHandler()
        data_processor.run_pipeline()
        print(f"Step 1: Data processing completed and saved to {step1_output_path}")
    else:
        print(f"Step 1: Data already processed and saved to {step1_output_path}")
    

    print("Initializing LLM services")
    summarizer = LocalSummarizer(config.SUMMARIZATION_MODEL,
                                 config.USE_4BIT_QUANTIZATION)
    embedder = LocalEmbedder(model_name = config.EMBEDDING_MODEL)
    
    print("LLM services initialized")

    item_extractor = ItemFeatureExtractor(embedder)

    for domain_key in config.DOMAINS.keys():
        output_path = os.path.join(config.HANDLE_DATA_DIR, f"itm_emb_np_{domain_key}.pkl")
        if not os.path.exists(output_path):
            item_extractor.run_for_domain(domain_key)
        else:
            print(f"Item embedding already extracted and saved to {output_path}")
    print("Step 2 finished")
    
    print("Running step3 User Profiling ")

    step3_output_path = os.path.join(config.HANDLE_DATA_DIR, "usr_profile_emb_final.pkl")
    if not os.path.exists(step3_output_path):
        user_profiler = UserProfiler(embedder = embedder,summarizer = summarizer)
        user_profiler.run_pipeline()
        print("Step 3 finished")
    else:
        print(f"Step 3: User profile already processed and saved to {step3_output_path}")

    reducer = DimensionalityReducer()
    reducer.run_pipeline()
    print("Step 4 finished")

    print("=" * 50)
    print("=== Full Pipeline Finished Successfully ===")
    print("=" * 50)

if __name__ == "__main__":
    login(HF_TOKEN)
    main()