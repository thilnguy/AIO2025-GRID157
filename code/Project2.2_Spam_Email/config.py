class Config:
    Data_Path = "data/2cls_spam_text_cls.csv"
    FAISS_Index_Path = "assets/faiss_index.bin"
    TRAIN_METADATA_PATH = "assets/train_metadata.csv"
    TEST_METADATA_PATH = "assets/test_metadata.csv"
    EVALUATION_METADATA_PATH = "assets/evaluation_results.json"
    MISPREDS_PATH = "errors/mispredictions.json"
    LOG_FILE = "logs/app.log"

    MODEL_NAME = "intfloat/multilingual-e5-base"
    QUERY_PREFIX = "query: "
    PASSAGE_PREFIX = "passage: "
    MAX_LENGTH = 512
    BATCH_SIZE = 32

    DEFAULT_K = 5

    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    LABEL2ID = {"ham": 0, "spam": 1}
    ID2LABEL = {0: "ham", 1: "spam"}