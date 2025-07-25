import numpy as np
from utils.preprocess import preprocess_text
from config import Config

class KNNClassifier:
    def __init__(self, vector_store, k=None):
        self.vector_store = vector_store
        self.k = k if k is not None else Config.DEFAULT_K

    def predict(self, query_text:str) -> dict:
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
        query = preprocess_text(query_text)
        emb = self.vector_store.embedder.encode(query, is_query=True).astype("float32")
        similarites, indices = self.vector_store.index.search(emb, self.k)
        
        labels = self.vector_store.train_metadata.iloc[indices[0]]['Category'].tolist()
        scores = similarites[0].tolist()

        predicted_label, counts = np.unique(labels, return_counts=True)
        final_label = predicted_label[np.argmax(counts)]

        neighbors = [
            {
                "message": self.vector_store.train_metadata.iloc[i]['Message'],
                "label": lbl,
                "score": float(sim)
            }
            for i, lbl, sim in zip(indices[0], labels, scores)
        ]

        return {
            "prediction": final_label,
            "neighbors": neighbors
        }

