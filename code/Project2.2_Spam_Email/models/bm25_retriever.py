from rank_bm25 import BM25Okapi
from utils.preprocess import preprocess_text, word_tokenize
from config import Config
import numpy as np 

class BM25Retriever:
    def __init__(self, messages, labels=None):
        """
        Initialize the BM25 retriever with a list of documents.
        
        Args:
            documents (list of str): List of documents to index.
        """
        self.messages = messages 
        self.labels = labels
        tokenized = [word_tokenize(preprocess_text(mess)) for  mess in messages]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query, BM25_TOP_K=None):
        """
        Retrieve the top-k documents relevant to the query.
        
        Args:
            query (str): The query string.
            top_k (int): Number of top documents to return.
        
        Returns:
            list: List document index.
        """
        bm25_top_k = BM25_TOP_K if BM25_TOP_K is not None else Config.BM25_TOP_K
        query_tokens = word_tokenize(preprocess_text(query))
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1][:bm25_top_k]
        
        return ranked_indices
    
    def retrieve_with_labels(self, query, BM25_TOP_K=None):
        indices = self.retrieve(query, BM25_TOP_K)
        return [self.labels[i] for i in indices]