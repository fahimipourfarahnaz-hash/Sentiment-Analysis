import streamlit as st
from transformers import pipeline

# Load the sentiment analysis model
# We'll use a small, efficient model for this example.
# 'distilbert-base-uncased-finetuned-sst-2-english' is a good choice.
@st.cache_resource
def get_sentiment_analyzer():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Initialize the sentiment analyzer
sentiment_analyzer = get_sentiment_analyzer()

# Set up the Streamlit app layout and title
st.set_page_config(page_title="Text Sentiment Analyzer", layout="centered")
st.title("Text Sentiment Analyzer")

# User input text box
st.markdown("### Analyze the following text and classify the overall sentiment.")
user_input = st.text_area("Enter your text here:", height=150)

# Main logic for sentiment analysis
if st.button("Analyze"):
    if user_input:
        # Perform sentiment analysis
        result = sentiment_analyzer(user_input)[0]

        # Get the sentiment label and confidence score
        label = result['label']
        score = result['score']

        # Determine the emoji for the label
        if label == 'POSITIVE':
            emoji = '😊'
        elif label == 'NEGATIVE':
            emoji = '😞'
        else:
            emoji = '😐' # Default for neutral, though this model gives pos/neg

        # Display the results
        st.markdown("---")
        st.markdown(f"**Sentiment Label:** {label} {emoji}")
        st.markdown(f"**Confidence Score:** {score:.4f}")
    else:
        st.warning("Please enter some text to analyze.")

# Add a section to explain how it works
st.markdown("---")
st.markdown("### How it Works")
st.markdown(
    """
    This app uses a pre-trained machine learning model from the **Hugging Face Transformers** library.
    Specifically, it employs a **DistilBERT** model that has been fine-tuned for sentiment analysis on
    the SST-2 dataset. The model processes the text and outputs a classification (`POSITIVE` or `NEGATIVE`)
    along with a **confidence score** representing how sure the model is of its prediction.
    """
)
st.image("https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/sst2_sentiment_example.png", caption="Example of sentiment analysis on the SST-2 dataset", use_column_width=True)
