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
    evaluator.evaluate_multinomial_naive_bayes()
    evaluator.evaluate_gaussian_naive_bayes()

    # Save trained models for Streamlit app
    evaluator.multinomial_nb.save_model("assets/multinomial_nb_model.pkl")
    evaluator.gaussian_nb.save_model("assets/gaussian_nb_model.pkl")

    # Save evaluation results
    evaluator.save_results()
    evaluator.print_summary()

    # Save mispredictions
    evaluator._save_mispredictions()
