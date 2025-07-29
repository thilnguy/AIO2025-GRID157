import numpy as np
import faiss
from utils.preprocess import preprocess_text
from config import Config
from models.bm25_retriever import BM25Retriever
from models.vector_store import VectorStore
from utils.logger import setup_logger
from collections import Counter

logger = setup_logger("classifier")

class HybridKNNClassifier:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.messages = vector_store.train_metadata["Message"].tolist()
        self.labels = vector_store.train_metadata["Category"].tolist()
        
        # Build BM25 retriever for training metadata
        self.bm25 = BM25Retriever(self.messages, self.labels)
        

    def predict(self, query_text:str, k=None) -> dict:
        """
        Predict the label for a given query using KNN.
        
        Args:
            query (str): The input text to classify.
        
        Returns:
            dict {
                str: Predicted label,
                neighbors: List of nearest neighbors' labels
            }
        """
        k = k if k is not None else Config.DEFAULT_K

        # Retrieve bm25-top-k documents using BM25
        bm25_indices = self.bm25.retrieve(query_text)
        logger.info(f"BM25 retrieved {len(bm25_indices)} candidates.")
       
        # GET candidate messages and labels
        candidate_messages = [self.messages[i] for i in bm25_indices]
        candidate_labels = [self.labels[i] for i in bm25_indices]

        # embedding for bm25_top-k candidates
        candidate_embeddings = self.vector_store.embeddings[bm25_indices]
        faiss.normalize_L2(candidate_embeddings)

        # similarity with query
        query_embedding = self.vector_store.embedder.encode(query_text, is_query=True).astype("float32")
        faiss.normalize_L2(query_embedding)

        similarities = candidate_embeddings @ query_embedding.T
        similarities = similarities.flatten()

        # Top-KNN candidates in top-k BM25 candidates
        knn_indices_in_candidate = np.argsort(similarities)[::-1][:k]
        
        top_similarities = similarities[knn_indices_in_candidate]
        final_labels = [candidate_labels[i] for i in knn_indices_in_candidate]
    
        prediction = Counter(final_labels).most_common(1)[0][0]

        neighbors = [
            {
                "message": candidate_messages[idx],
                "label": lbl,
                "similarity": float(sim)
            }
            for idx, lbl, sim in zip(knn_indices_in_candidate, final_labels, top_similarities)
        ]

        return {
            "prediction": prediction,
            "neighbors": neighbors
        }