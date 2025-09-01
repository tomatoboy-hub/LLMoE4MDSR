import os 
import config
from llm_servcies import LocalSummarizer, LocalEmbedder
from step1_data_processor import AmazonHandler
from step2_feature_extractor import ItemFeatureExtractor
from ste3_user_profiler import UserProfiler

def main():
    print("=" * 50)
    print("Starting data processing pipeline")
    print("=" * 50)

    handler = AmazonHandler()
    handler.run_pipeline()

    item_feature_extractor = ItemFeatureExtractor()
    item_feature_extractor.run_pipeline()

    user_profiler = UserProfiler()
    user_profiler.run_pipeline()

    print("Data processing pipeline completed successfully")

    step1_output_path = os.path.join(config.HANDLE_DATA_DIR, config.INTER_FILE_NAME)
    if not os.path.exsits(step1_output_path):
        data_processor = AmazonHandler()
        data_processor.run_pipeline()
        print(f"Step 1: Data processing completed and saved to {step1_output_path}")
    else:
        print(f"Step 1: Data already processed and saved to {step1_output_path}")
    

    print("Initiaalizing LLM services")
    summarizer = LocalSummarizer(config.SUMMARIZATION_MODEL,
                                 config.USE_4BIT_QUANTIZATION)
    embedder = LocalEmbedder(model_name = config.EMBEDDING_MODEL)
    
    print("LLM services initialized")

    item_extractor = ItemFeatureExtractor(embedder)

    for domain_key in config.DOMAINS.keys():
        output_path = os.path.join(config.HANDLE_DATA_DIR, f"itm_emb_np_{domain_key}.pkl")
        if not os.path.exists(output_path):
            item_extractor.run_pipeline(domain_key)
        else:
            print(f"Item embedding already extracted and saved to {output_path}")
    print("Step 2 finished")
    
    print("Running step3 User Profiling ")