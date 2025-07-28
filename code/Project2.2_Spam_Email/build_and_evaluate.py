from models.vector_store import VectorStore
from models.evaluator import ModelEvaluator
from config import Config

if __name__ == "__main__":
    # Initialize the vector store
    vector_store = VectorStore(batch_size=Config.BATCH_SIZE)
    
    # Build the vector store from CSV data
    vector_store.build_from_csv()
    
    # Save the FAISS index
    vector_store.save()
    
    # Initialize the evaluator
    evaluator = ModelEvaluator(vector_store)
    
    # Evaluate the model
    evaluator.evaluate_embedding_only()
    evaluator.evaluate_bm25_only()
    evaluator.evaluate_hybrid()

    # Save evaluation results
    evaluator.save_results()
    evaluator.print_summary()
    
    # Save mispredictions
    #y_preds = evaluator.hybrid_knn.predict(evaluator.test_df['Message'].tolist())

   # evaluator._save_mispredictions()
