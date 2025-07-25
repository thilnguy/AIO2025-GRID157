import pandas as pd
from config import Config
from utils.logger import setup_logger
import json
import os 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from models.knn_classifier import KNNClassifier

logger = setup_logger("evaluator")

class ModelEvaluator:
    def __init__(self, vector_store, test_data_path=Config.TEST_METADATA_PATH, k=None):
        self.vector_store = vector_store
        self.test_df = pd.read_csv(test_data_path)
        self.k = k if k is not None else Config.DEFAULT_K
        self.knn = KNNClassifier(vector_store, k=self.k)

    def evaluate(self):
        """
        Evaluate the model on the test dataset.
        
        Returns:
            dict: Evaluation metrics including accuracy, precision, recall, and F1-score.
        """
        # Load test data
        logger.info(f"evaluating model with k ={self.k}")

        y_true = self.test_df['Category'].map(Config.LABEL2ID).values
        y_preds = []

        for message in self.test_df['Message']:
            result = self.knn.predict(message)
            pred_id = Config.LABEL2ID[result['prediction']]
            y_preds.append(pred_id)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_preds, average='weighted')

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "k": self.k
        }

        with open(Config.EVALUATION_METADATA_PATH, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.save_mispreds(y_preds)
        logger.info(f"Evaluation Metrics: {metrics}")
        return metrics
    
    def save_mispreds(self, y_preds):
        """
        Save mispredictions to a JSON file.
        
        Args:
            y_preds (list): List of predicted labels.
        """
        mispreds = []
        for idx, row in self.test_df.iterrows():
            if y_preds[idx] == Config.LABEL2ID[row['Category']]:
                continue
            result = self.knn.predict(row['Message'])
            mispreds.append({
                "message": row['Message'],
                "true_label": row['Category'],
                "predicted_label": result['prediction'],
                "neighbors": result['neighbors']
            })
        
        os.makedirs(os.path.dirname(Config.MISPREDS_PATH), exist_ok=True)
        with open(Config.MISPREDS_PATH, 'w') as f:
            json.dump(mispreds, f, indent=2)
        print(f"Saved {len(mispreds)} mispredictions to {Config.MISPREDS_PATH}")

