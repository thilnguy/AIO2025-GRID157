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
    evaluator = ModelEvaluator(vector_store, k=Config.DEFAULT_K)
    
    # Evaluate the model
    metrics = evaluator.evaluate()
    
    print(f"Evaluation Metrics: {metrics}")