import streamlit as st
from transformers import pipeline
from dotenv import load_dotenv, find_dotenv

# Load environment variables from a .env file (if needed)
load_dotenv(find_dotenv())

# Set page configuration
st.set_page_config(page_title="Diy's Text Summarizer", layout="wide")

# Cache the pipeline for faster performance
@st.cache_resource(show_spinner=False)
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# App Title and Description
st.title("Text Summarization App")
st.write("""
This application uses Hugging Face's BART model to summarize your text.
Enter your text in the box below or upload a text file, and click on **Summarize**.
""")

# Text input area
text_input = st.text_area("Enter your text here:", height=200)

# File uploader (optional)
uploaded_file = st.file_uploader("...or upload a text file", type=["txt"])
if uploaded_file is not None:
    # Read file as string
    file_text = uploaded_file.read().decode("utf-8")
    st.info("File content loaded into the text area.")
    text_input = file_text  # Overwrite text_input with file content

# Summarize button
if st.button("Summarize"):
    if text_input.strip() == "":
        st.warning("Please provide some text for summarization.")
    else:
        with st.spinner("Summarizing..."):
            # Adjust max_length and min_length as needed
            summary_output = summarizer(text_input, max_length=200, min_length=30, do_sample=False)
            summarized_text = summary_output[0]['summary_text']
        st.subheader("Summary")
        st.write(summarized_text)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using [Streamlit](https://streamlit.io/) and [Transformers](https://huggingface.co/transformers/)")
