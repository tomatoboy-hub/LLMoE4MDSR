RAW_DATA_DIR = "./raw"
HANDLE_DATA_DIR = "./handled"

DOMAINS = {
    "0":"Clothing_Shoes_and_Jewelry",
    "1":"Sports_and_Outdoors",
    "2":"AMAZON_FASHION"
}

USER_CORE = 5
ITEM_CORE = 3
INTER_FILE_NAME  = "cloth_sport_fashion.pkl"

TIME_MIN = 1514736000  # 2018-01-01 00:00:00 GMT
TIME_MAX = 1577808000  # 2020-01-01 00:00:00 GMT

ITEM_PROMPT_TEMPLATE = "The {domain_type} item has following attributes: \n name is <TITLE>; brand is <BRAND>; price is <PRICE>."

NUM_CLUSTERS = 10
USER_PROFILE_PROMPTS = {
    "analyzer": "Assume you are a consumer who is shopping online. You have shown interests in following commdities:\n {}. \n\n The commodities are segmented by '\n'. \n\n Please conclude it not beyond 50 words. Do not only evaluate one specfic commodity but illustrate the interests overall.",
    "summarizer": "Assume you are an consumer and there are preference demonstrations from several aspects are as follows:\n {}. \n\n Please illustrate your preference with less than 100 words"
}


SUMMARIZATION_MODEL = "google/gemma-3-270m"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


USE_4BIT_QUANTIZATION = True 

PCA_TARGET_DIMENSION = 128