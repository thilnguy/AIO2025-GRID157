"""
Example usage of the Naive Bayes classifier for spam/ham email classification.

This script demonstrates how to use the NaiveBayesClassifier independently.
"""

from models.vector_store import VectorStore
from models.naive_bayes import NaiveBayesClassifier

def main():
    """Demonstrate Naive Bayes classifier usage."""

    # Load the vector store (make sure to run build_and_evaluate.py first)
    print("Loading vector store...")
    try:
        vector_store = VectorStore().load()
        print(f"✅ Vector store loaded with {len(vector_store.train_metadata)} training examples")
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        print("Please run 'python build_and_evaluate.py' first to build the vector store.")
        return

    # Initialize Naive Bayes classifier
    print("\nInitializing Naive Bayes classifier...")
    nb_classifier = NaiveBayesClassifier(vector_store, alpha=1.0)

    # Display model information
    model_info = nb_classifier.get_model_info()
    print(f"\n📊 Model Information:")
    print(f"   - Model Type: {model_info['model_type']}")
    print(f"   - Training Documents: {model_info['num_training_docs']}")
    print(f"   - Vocabulary Size: {model_info['vocabulary_size']}")
    print(f"   - Smoothing Alpha: {model_info['smoothing_alpha']}")
    print(f"   - Class Distribution: {model_info['class_counts']}")

    # Example predictions
    test_messages = [
        "Congratulations! You've won $1000! Click here to claim your prize!",
        "Hi, let's meet for coffee tomorrow at 3pm. How does that sound?",
        "URGENT! Your account will be suspended. Click here immediately!",
        "Thank you for your email. I'll get back to you soon.",
        "FREE MONEY!!! No credit check required. Apply now!!!",
        "Can you please send me the project report by Friday?"
    ]

    print(f"\n🔍 Example Predictions:")
    print("=" * 60)

    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. Message: \"{message[:50]}{'...' if len(message) > 50 else ''}\"")

        # Make prediction
        result = nb_classifier.predict(message, return_probabilities=True)

        # Display results
        prediction = result['prediction']
        confidence = result['confidence']
        probabilities = result['probabilities']

        emoji = "🔴" if prediction == "spam" else "🟢"
        print(f"   {emoji} Prediction: {prediction.upper()} (Confidence: {confidence:.3f})")
        print(f"   📊 Probabilities: Ham={probabilities['ham']:.3f}, Spam={probabilities['spam']:.3f}")

        # Show most informative words
        if result['neighbors']:
            print(f"   🔍 Key words: {', '.join([word['message'].split(':')[1].strip().split()[0] for word in result['neighbors'][:3]])}")

if __name__ == "__main__":
    main()