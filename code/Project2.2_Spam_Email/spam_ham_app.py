import streamlit as st
from models.vector_store import VectorStore
from models.knn_classifier import HybridKNNClassifier
from config import Config
import os


@st.cache_resource
def load_hybrid_classifier():
    try:
        vs = VectorStore().load()
        return HybridKNNClassifier(vs)
    except Exception as e:
        st.error(f"❌ Không thể tải model: {e}")
        st.info("Hãy chạy `python build_vector_store.py` trước.")
        return None

st.set_page_config(
    page_title="📬 Spam Email Classification",
    layout="centered"
)

def clear_text():
    st.session_state.email_text = ""
    st.rerun()

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
        options=["Hybrid (BM25 + Embedding)", "Naive Bayes Classifier"]
    )
    
    # Chọn K (chỉ áp dụng cho Hybrid)
    if model_choice == "Hybrid (BM25 + Embedding)":
        k = st.slider("Number Neighbor K for KNN_model", min_value=1, max_value=10, value=Config.DEFAULT_K)
    else:
        k = 5

# === NHẬP EMAIL ===
email_text = st.text_area(
    "email:",
    height=150,
    placeholder="Hello"
)

# === NÚT PHÂN LOẠI ===
if classify_clicked:
    if not email_text.strip():
        st.warning("⚠️ Type email!")
    else:
        if model_choice == "Naive Bayes Classifier":
            # thêm code vào đây
            st.info("🧪 Đây là mô hình mẫu (Naive Bayes)")
            st.success("**Kết quả: SPAM**")
            st.markdown("""
            **Lý do (mẫu):**  
            - Từ khóa:  
            - Xác suất cao là spam theo thống kê.
            """)

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