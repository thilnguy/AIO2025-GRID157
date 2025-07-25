from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch
from config import Config
from utils.logger import setup_logger

logger = setup_logger("embedder")

class E5Embedder:
    def __init__(self):
        self.model = AutoModel.from_pretrained(Config.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

        # check GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

    
    def average_pooling(self, last_hidden_state, attention_mask):
        masked_hidden = last_hidden_state.masked_fill(~attention_mask.unsqueeze(-1).bool(), 0)
        return masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

    def encode(self, texts, is_query=False, max_length=None):
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts (list of str): List of texts to encode.
        
        Returns:
            np.ndarray: Array of embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        max_length = max_length if max_length is not None else Config.MAX_LENGTH
        prefix = Config.QUERY_PREFIX if is_query else Config.PASSAGE_PREFIX
        prefixed = [prefix + text for text in texts]
        # Encode the texts using the model
        inputs = self.tokenizer(
            prefixed, 
            padding=True, 
            truncation=True, 
            return_tensors='pt', 
            max_length=max_length
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            embeddings = self.average_pooling(last_hidden_state, attention_mask)
        # Convert embeddings to numpy array
        return embeddings.cpu().numpy().astype("float32")
    