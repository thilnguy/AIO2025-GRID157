import faiss 
import pandas as pd
import numpy as np
from models.embedder import E5Embedder
from config import Config
from utils.logger import setup_logger
from utils.preprocess import preprocess_text
from sklearn.model_selection import train_test_split


logger = setup_logger("vector_store")

def batch_geneator(df, batch_size=Config.BATCH_SIZE):
    """Generator to yield batches of data."""
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i + batch_size]

class VectorStore:
    def __init__(self, dim=768, batch_size = None):
        self.batch_size = batch_size if batch_size is not None else Config.BATCH_SIZE
        self.dim = dim
        self.embedder = E5Embedder()
        self.embeddings = None
        self.train_metadata = None

        self.index = faiss.IndexFlatL2(dim)
        if faiss.get_num_gpus() > 0:
            logger.info("Using GPU for FAISS index.")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_all_gpus(res, 0, self.index)
        else:
            logger.info("Using CPU for FAISS index.")
    

    def build_from_csv(self, data_path=Config.Data_Path):
        """Build the vector store from a CSV file."""
        df = pd.read_csv(data_path)
        df['cleaned_text'] = df['Message'].apply(preprocess_text)
        
        #split the data into train and test
        train_df, test_df = train_test_split(df, 
                            test_size=Config.TEST_SIZE, 
                            random_state=Config.RANDOM_STATE,
                            stratify=df['Category'])
        

        #train_df[["Category","Message"]].to_csv(Config.TRAIN_METADATA_PATH, index=False)
        test_df[["Category","Message"]].to_csv(Config.TEST_METADATA_PATH, index=False)

        # batch processing
        total = len(train_df)
        processed = 0
        all_embeddings = []

        for batch_df in batch_geneator(train_df, self.batch_size):
            texts = batch_df['cleaned_text'].tolist()
            embs = self.embedder.encode(texts, is_query=False)
            all_embeddings.append(embs)

            processed += len(batch_df)
            logger.info(f"Processed {processed}/{total} rows.")
        
        self.embeddings = np.vstack(all_embeddings)
        np.save(Config.EMBEDDINGS_DIR, self.embeddings)

        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

        #save the index and metadata
        self.train_metadata = train_df
        self.train_metadata.to_pickle(Config.TRAIN_METADATA_PATH)
        logger.info(f"Vector store built with {len(train_df)} vectors.")
    
    def save(self, path=Config.FAISS_Index_Path):
        """Save the FAISS index to a file."""
        faiss.write_index(self.index, path)
        logger.info(f"FAISS index saved to {path}.")

    def load(self, path=Config.FAISS_Index_Path):
        """Load the FAISS index from a file."""
        self.index = faiss.read_index(path)
        logger.info(f"FAISS index loaded from {path}.")
        # USE GPU if available
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_all_gpus(res, 0, self.index)
            logger.info("FAISS index moved to GPU.")
        else:
            logger.info("FAISS index is on CPU.")
        
        self.train_metadata = pd.read_pickle(Config.TRAIN_METADATA_PATH)
        self.embeddings = np.load(Config.EMBEDDINGS_DIR)
        
        return self
    