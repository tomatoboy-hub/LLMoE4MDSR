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

ITEM_PROMPT_TEMPLATE = "The fashion item has following attributes: \n name is <TITLE>; brand is <BRAND>; price is <PRICE>."

NUM_CLUSTERS = 10
USER_PROFILE_PROMPTS = {
    "analyzer": "Assume you are a consumer who is shopping online. You have shown interests in following commdities:\n {}. \n\n The commodities are segmented by '\n'. \n\n Please conclude it not beyond 50 words. Do not only evaluate one specfic commodity but illustrate the interests overall.",
    "summarizer": "Assume you are an consumer and there are preference demonstrations from several aspects are as follows:\n {}. \n\n Please illustrate your preference with less than 100 words"
}


SUMMARIZATION_MODEL = "meta-llama/Llama-3-8B-Instruct"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


USE_4BIT_QUANTIZATION = True 


