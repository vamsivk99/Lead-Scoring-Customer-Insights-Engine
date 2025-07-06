import streamlit as st
import os
from src.rag_pipeline import load_rag_pipeline, ask_question
from src.lead_scoring import get_lead_scorer, score_lead
import glob
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Intelligent Lead Insights Engine",
    page_icon="💡",
    layout="wide"
)

# --- API Key Input ---
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("OpenRouter API Key", type="password")

# --- Caching Functions ---
@st.cache_resource
def load_pipeline_cached(api_key_provided):
    """Cached function to load the RAG pipeline."""
    try:
        return load_rag_pipeline(api_key_provided)
    except (ValueError, FileNotFoundError) as e:
        st.error(f"Failed to load RAG pipeline: {e}")
        st.warning("Please ensure the vector store exists. Run `python src/create_vector_store.py` if needed.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred loading the RAG pipeline: {e}")
        return None


@st.cache_resource
def load_scorer_cached(api_key_provided):
    """Cached function to load the lead scoring pipeline."""
    try:
        return get_lead_scorer(api_key_provided)
    except Exception as e:
        st.error(f"Failed to load Lead Scorer: {e}")
        return None

# --- Main Application ---
st.title("Intelligent Lead Insights Engine")

if not api_key:
    st.info("Please enter your OpenRouter API Key in the sidebar to begin.")
    st.stop()
else:
    st.sidebar.success("✅ OpenAI API Key Loaded")

# Load Pipelines
rag_chain = load_pipeline_cached(api_key)
scorer_chain = load_scorer_cached(api_key)

if not rag_chain or not scorer_chain:
    st.warning("One or more AI pipelines could not be loaded. Please check the errors above.")
    st.stop()

# --- Main UI Tabs ---
tab1, tab2 = st.tabs(["Ask Questions", "Score Leads"])

# 1. Querying Interface
with tab1:
    st.header("Ask Questions About Your Documents")
    query = st.text_input("Enter your query:", placeholder="e.g., What are the terms for a commercial real estate loan?")
    if st.button("Get Answer"):
        with st.spinner("Searching for answers..."):
            result = ask_question(query, rag_chain)
            st.success("Answer")
            st.write(result["answer"])
            with st.expander("Show Source Documents"):
                for doc in result["context"]:
                    st.info(f"Source: {os.path.basename(doc.metadata.get('source', 'N/A'))}")

# 2. Lead Scoring Interface
with tab2:
    st.header("Score Your Leads")
    if st.button("Score All Documents"):
        document_files = glob.glob("mock_docs/*.txt")
        if not document_files:
            st.warning("No documents found in 'mock_docs'. Please generate them first.")
        else:
            with st.spinner("Scoring all lead documents... This may take a moment."):
                all_scores = [score_lead(doc, scorer_chain) for doc in document_files]
                sorted_scores = sorted([s for s in all_scores if 'score' in s], key=lambda x: x['score'], reverse=True)
                st.dataframe(sorted_scores, use_container_width=True) 