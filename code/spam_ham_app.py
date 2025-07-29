import streamlit as st

# Page config
st.set_page_config(
    page_title="Spam or ham - email classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

def process_selection(choice):
    """
    Process the selected choice
    :param choice (str): The chosen option
    return: a tuple of classified email type and its confidence score
    """
    if choice == "Naive Bayes Classifier":
        # TODO: Update the output of NBC here
        return "spam", "score1"
    elif choice == "Vector database":
        # TODO: Update the output of vector database here
        return "ham", "score2"

    
def main():
    
    # setup session_state
    if "email_text" not in st.session_state:
        st.session_state.email_text = ""
    
    if st.session_state.get("clear_input"):
        st.session_state.email_text = ""
        st.session_state.clear_input = False

    # Header
    st.markdown('<h1 class="main-header"> Spam or ham - email classifier </h1>', unsafe_allow_html=True)
    st.markdown("Classify the email as spam or ham using naive-bayes classification or vector database")
    
    # Textbox
    st.text_input("Enter the email's content here ...", key="email_text")
    
    # Doing the classification only if the content of email is entered
    if st.session_state.email_text:
        selected_options = st.radio("Choose one method to classify:", ["Naive Bayes Classifier", "Vector database"])
        
        # Output the result
        result, confidence_score = process_selection(selected_options)
        
        st.success(f"This email is classified as: {result.upper()}")
        st.info(f"Confidence score is: {confidence_score}")
    
        st.button("Clear", key="clear_input")
    
if __name__ == "__main__":
    main()