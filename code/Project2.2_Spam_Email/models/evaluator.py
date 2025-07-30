import pandas as pd
from config import Config
from utils.logger import setup_logger
import json
import os
import faiss
from utils.preprocess import preprocess_text
from collections import Counter
from models.bm25_retriever import BM25Retriever
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from models.knn_classifier import HybridKNNClassifier
import time

logger = setup_logger("evaluator")

class ModelEvaluator:
    def __init__(self, vector_store, test_data_path=Config.TEST_METADATA_PATH, k=None):
        self.vector_store = vector_store
        self.test_df = pd.read_csv(test_data_path)
        self.k = k if k is not None else Config.DEFAULT_K
        self.hybrid_knn = HybridKNNClassifier(vector_store)
        self.bm25_retriever = BM25Retriever(messages=vector_store.train_metadata["Message"].tolist(),
                                            labels=vector_store.train_metadata["Category"].tolist())
        self.results = {}


    def evaluate_embedding_only(self, k=Config.DEFAULT_K):
        """
        Evaluate the embedding model on the test dataset.
        
        Returns:
            dict: Evaluation metrics including accuracy, precision, recall, and F1-score.
        """
        # Load test data
        logger.info(f"evaluating model with k = %d",self.k)
        embedder = self.vector_store.embedder

        total_time = 0.0

        y_true = self.test_df['Category'].map(Config.LABEL2ID).values
        y_preds = []

        for message in self.test_df['Message']:
            start_time = time.time()
            
            message = preprocess_text(message)
            query_embedding = embedder.encode(message, is_query=True).astype("float32")
            faiss.normalize_L2(query_embedding)

            # find KNN in vector store
            similarities, indices = self.vector_store.index.search(query_embedding, k=self.k)
            neighbor_labels = self.vector_store.train_metadata.iloc[indices[0]]['Category'].tolist()

            # majority voting
            pred_label = Counter(neighbor_labels).most_common(1)[0][0]
            pred_label = Config.LABEL2ID[pred_label]

            y_preds.append(pred_label)
            total_time += time.time() - start_time
        
        avg_time = total_time / len(self.test_df)

        metrics = self._compute_metrics(y_true, y_preds, "Embedding", avg_time)
        self.results['embedding'] = metrics
        return metrics

    def evaluate_bm25_only(self):
        """
        Evaluate the BM25 retriever on the test dataset.
        
        Returns:
            dict: Evaluation metrics including accuracy, precision, recall, and F1-score.
        """
        y_true = self.test_df['Category'].map(Config.LABEL2ID).values
        y_preds = []
        total_time = 0.0

        for message in self.test_df['Message']:
            start_time = time.time()
            retrieved_labels = self.bm25_retriever.retrieve_with_labels(message, BM25_TOP_K=Config.DEFAULT_K)

            #majority voting
            pred_label = Counter(retrieved_labels).most_common(1)[0][0]
            pred_id = Config.LABEL2ID[pred_label]
            y_preds.append(pred_id)

            total_time +=time.time() - start_time
        
        avg_time = total_time / len(self.test_df)
        metrics = self._compute_metrics(y_true, y_preds, "BM25", avg_time)
        self.results['bm25'] = metrics
        return metrics

    def evaluate_hybrid(self):
        """
        Evaluate the hybrid model (BM25 + Embedding) on the test dataset.
        Also saves mispredictions with neighbor context for analysis.
        
        Returns:
            dict: Evaluation metrics including accuracy, precision, recall, and F1-score.
        """
        y_true = self.test_df['Category'].map(Config.LABEL2ID).values
        y_preds = []
        self.mispredictions = []
        total_time = 0.0

        for idx, row in self.test_df.iterrows():
            start_time = time.time()
            message = row['Message']
            true_label = row['Category']  # 'spam' or 'ham'
        
            result = self.hybrid_knn.predict(message)
            pred_label = result['prediction']
            pred_id = Config.LABEL2ID[pred_label]
            
          
            total_time += time.time() - start_time

            y_preds.append(pred_id)

            if true_label != pred_label:
                self.mispredictions.append({
                    "message": message,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "neighbors": result["neighbors"],  # {message, label, similarity}
                    "index_in_test": int(idx)
                })

        # metrics
        avg_time = total_time / len(self.test_df)
        metrics = self._compute_metrics(y_true, y_preds, "Hybrid", avg_time)
        self.results['hybrid'] = metrics

        # Save mispredictions
        self._save_mispredictions()

        return metrics
    
    def _compute_metrics(self, y_true, y_preds, model_name, avg_inference_time):
        """
        Compute evaluation metrics.
        
        Args:
            y_true (np.ndarray): True labels.
            y_preds (list): Predicted labels.
            model_name (str): Name of the model being evaluated.
        
        Returns:
            dict: Evaluation metrics including accuracy, precision, recall, and F1-score.
        """
        accuracy = accuracy_score(y_true, y_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_preds, average='weighted')

        cm = confusion_matrix(y_true, y_preds)
        self._plot_confusion_matrix(cm, model_name)


        metrics = {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_inference_time': float(avg_inference_time)
        }
        
        logger.info(f"Metrics for %s: %s", model_name, metrics)
        return metrics
    
    def _plot_confusion_matrix(self, cm, model_name):
        """
        Plot and save the confusion matrix.
        
        Args:
            cm (np.ndarray): Confusion matrix.
            model_name (str): Name of the model.s
        """
        cm_dir = Config.CM_DIR
        os.makedirs(cm_dir, exist_ok=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=Config.ID2LABEL.values(), yticklabels=Config.ID2LABEL.values())
        plt.title(f'Confusion Matrix for {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(f"{Config.CM_DIR}/{model_name}_confusion_matrix.png")
        plt.close()


    def _save_mispredictions(self):
        """
        Save mispredictions to a JSON file.
        """
        mispred_path = Config.MISPREDS_PATH
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(mispred_path), exist_ok=True)
        
        # Lưu file JSON
        with open(mispred_path, 'w', encoding='utf-8') as f:
            json.dump(self.mispredictions, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Save %d misprediction in %s", len(self.mispredictions), mispred_path)


    def save_results(self):
        """
        Save evaluation results to a JSON file.
        """
        results = {
            "results": self.results,
            "config": {
                "model_name": Config.MODEL_NAME,
                "k": self.k,
                "test_size": Config.TEST_SIZE,
                "random_state": Config.RANDOM_STATE
            }
        }
        os.makedirs(os.path.dirname(Config.EVALUATION_METADATA_PATH), exist_ok=True)
        with open(Config.EVALUATION_METADATA_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to %s", Config.EVALUATION_METADATA_PATH)

    def print_summary(self):
        """
        Print a summary of the evaluation results.
        """
        logger.info("Evaluation Summary:")
        for model, metrics in self.results.items():
            logger.info("%s - Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1-Score: %.4f, Time: %.4f s",
                        model.upper(),
                        metrics['accuracy'],
                        metrics['precision'],
                        metrics['recall'],
                        metrics['f1_score'],
                        metrics['avg_inference_time'])