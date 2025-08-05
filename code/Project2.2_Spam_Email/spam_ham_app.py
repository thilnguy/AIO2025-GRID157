import streamlit as st
from models.vector_store import VectorStore
from models.knn_classifier import HybridKNNClassifier
from models.naive_bayes import MultinomialNaiveBayes, GaussianNaiveBayes
from config import Config
import os


@st.cache_resource
def load_hybrid_classifier():
    try:
        vs = VectorStore().load()
        return HybridKNNClassifier(vs)
    except Exception as e:
        st.error(f"❌ Can't load model: {e}")
        st.info("Run `python build_vector_store.py`.")
        return None

@st.cache_resource
def load_naive_bayes_classifier(model_type='multinomial', alpha=1.0, max_features=5000):
    try:
        model_path = f"assets/{model_type}_nb_model.pkl"

        # Try to load pre-trained model first
        if os.path.exists(model_path):
            if model_type == 'multinomial':
                return MultinomialNaiveBayes.load_model(model_path)
            else:
                return GaussianNaiveBayes.load_model(model_path)
        else:
            # Fall back to training from vector store
            st.warning(f"⚠️ No pre-trained {model_type} model found. Training from scratch...")
            vs = VectorStore().load()

            if model_type == 'multinomial':
                classifier = MultinomialNaiveBayes(vs, alpha=alpha)
            else:
                classifier = GaussianNaiveBayes(vs, alpha=alpha, max_features=max_features)

            # Save for future use
            classifier.save_model(model_path)
            return classifier
    except Exception as e:
        st.error(f"❌ Can't load {model_type} Naive Bayes model: {e}")
        st.info("Run `python build_and_evaluate.py` first.")
        return None

st.set_page_config(
    page_title="📬 Spam Email Classification",
    layout="centered"
)


if "my_text" not in st.session_state:
    st.session_state.my_text = ""

def clear_text():
    st.session_state.my_text = ""

st.title("📬 Email: Spam or Ham?")

col1, col2 = st.columns([1, 1])
with col1:
    classify_clicked = st.button("🔍 Submit")

with col2:
    st.button("🗑️ Refresh", on_click=clear_text)

with st.sidebar:
    st.header("⚙️ Config")


    model_choice = st.radio(
        "Select Model",
        options=["Hybrid (BM25 + Embedding)", "Multinomial Naive Bayes", "Gaussian Naive Bayes"]
    )

    # Show model availability status
    if model_choice in ["Multinomial Naive Bayes", "Gaussian Naive Bayes"]:
        nb_type = "multinomial" if "Multinomial" in model_choice else "gaussian"
        model_path = f"assets/{nb_type}_nb_model.pkl"
        is_available = os.path.exists(model_path)
        if is_available:
            st.success(f"✅ Pre-trained {model_choice} model available")
        else:
            st.warning(f"⚠️ {model_choice} model will be trained on first use")

    # Model-specific parameters
    if model_choice == "Hybrid (BM25 + Embedding)":
        k = st.slider("Number Neighbor K for KNN_model", min_value=1, max_value=10, value=Config.DEFAULT_K)
    elif model_choice == "Gaussian Naive Bayes":
        max_features = st.slider("Max TF-IDF Features", min_value=1000, max_value=10000, value=5000, step=500)
        alpha = st.slider("Smoothing Alpha", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    elif model_choice == "Multinomial Naive Bayes":
        alpha = st.slider("Laplace Smoothing Alpha", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    else:
        k = 5

# === EMAIL ===
email_text = st.text_area(
    "email:",
    height=150,
    placeholder="Hello",
    key="my_text",
    value=st.session_state.my_text
)

# === Button ===
if classify_clicked:
    if not email_text.strip():
        st.warning("⚠️ Type email!")
    else:
        if model_choice in ["Multinomial Naive Bayes", "Gaussian Naive Bayes"]:
            nb_type = "multinomial" if "Multinomial" in model_choice else "gaussian"
            if model_choice == "Gaussian Naive Bayes":
                classifier = load_naive_bayes_classifier(nb_type, alpha, max_features)
            else:
                classifier = load_naive_bayes_classifier(nb_type, alpha)
            if classifier is None:
                st.stop()

            with st.spinner(f"Processing with {model_choice}..."):
                try:
                    result = classifier.predict(email_text, return_probabilities=True)
                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    probabilities = result["probabilities"]
                    informative_words = result["neighbors"]

                    # Display result
                    emoji = "🔴" if prediction == "spam" else "🟢"
                    st.success(f"{emoji} **{prediction.upper()}** (Confidence: {confidence:.3f})")

                    # Show model info
                    model_info = classifier.get_model_info()
                    st.info(f"📊 Model: {model_info['model_type']} | "
                           f"Training docs: {model_info['num_training_docs']} | "
                           f"Alpha: {model_info['smoothing_alpha']}")
                    if 'vocabulary_size' in model_info:
                        st.info(f"Vocabulary Size: {model_info['vocabulary_size']}")
                    if 'n_features' in model_info:
                        st.info(f"Number of Features: {model_info['n_features']}")

                    # Show probabilities
                    st.markdown("### 📊 Class Probabilities:")
                    col1, col2 = st.columns(2)

                    with col1:
                        ham_prob = probabilities.get('ham', 0)
                        st.metric("🟢 HAM", f"{ham_prob:.3f}")
                        st.progress(ham_prob)

                    with col2:
                        spam_prob = probabilities.get('spam', 0)
                        st.metric("🔴 SPAM", f"{spam_prob:.3f}")
                        st.progress(spam_prob)

                    # Show most informative features
                    if informative_words:
                        st.markdown("### 🔍 Most Informative Features:")
                        for i, feature_info in enumerate(informative_words):
                            weight = feature_info["similarity"]
                            feature_text = feature_info["message"]
                            st.markdown(f"{i+1}. {feature_text}")

                except Exception as e:
                    st.error(f"Error during classification: {e}")

        elif model_choice == "Hybrid (BM25 + Embedding)":
            classifier = load_hybrid_classifier()
            if classifier is None:
                st.stop()

            with st.spinner("Process by Hybrid model..."):
                try:
                    result = classifier.predict(email_text, k=k)
                    prediction = result["prediction"]
                    neighbors = result["neighbors"]

                    # Hiển thị kết quả
                    st.success(f"✅ **{prediction.upper()}**")

                    st.markdown("### 🔍 Similarities Email:")
                    for i, nbr in enumerate(neighbors):
                        emoji = "🔴" if nbr["label"] == "spam" else "🟢"
                        st.markdown(
                            f"{i+1}. {emoji} **[{nbr['label'].upper()}]** "
                            f"(score: `{nbr['similarity']:.3f}`)\n"
                            f"> _{nbr['message']}_"
                        )
                except Exception as e:
                    st.error(f"error: {e}")