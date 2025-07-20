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
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Key Input ---
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("OpenRouter API Key", type="password", help="Enter your OpenRouter API key to use the AI models.")

with st.sidebar.expander("About the App"):
    st.write("""
        This application uses a Retrieval-Augmented Generation (RAG) pipeline 
        to analyze and score financial leads from a collection of documents.
        
        **Features:**
        - **Ask Questions**: Get answers from your documents.
        - **Score Leads**: Analyze documents to determine lead quality.
    """)

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
st.title("💡 Intelligent Lead Insights Engine")
st.markdown("Your AI-powered assistant for analyzing and scoring financial leads.")

if not api_key:
    st.info("Please enter your OpenRouter API Key in the sidebar to begin.")
    st.stop()
else:
    st.sidebar.success("✅ API Key Loaded")
    if st.sidebar.button("Clear Cache"):
        st.cache_resource.clear()
        st.rerun()

# Load Pipelines
rag_chain = load_pipeline_cached(api_key)
scorer_chain = load_scorer_cached(api_key)

if not rag_chain or not scorer_chain:
    st.warning("One or more AI pipelines could not be loaded. Please check the errors above.")
    st.stop()

# --- Main UI Tabs ---
tab1, tab2 = st.tabs(["❓ Ask Questions", "📊 Score Leads"])

# 1. Querying Interface
with tab1:
    st.header("Ask Questions About Your Documents")
    st.markdown("Use the RAG pipeline to find answers within your document set.")
    query = st.text_input("Enter your query:", placeholder="e.g., What are the terms for a commercial real estate loan?")
    
    if st.button("Get Answer"):
        if query:
            with st.spinner("🧠 Searching for answers..."):
                result = ask_question(query, rag_chain)
                st.success("Answer")
                st.write(result["answer"])
                with st.expander("📚 Show Source Documents"):
                    for doc in result["context"]:
                        st.info(f"**Source:** `{os.path.basename(doc.metadata.get('source', 'N/A'))}`")
                        st.text(doc.page_content[:500] + "...")
        else:
            st.warning("Please enter a query.")

# 2. Lead Scoring Interface
with tab2:
    st.header("Score Your Leads")
    st.markdown("Analyze individual documents or score all leads at once.")

    document_files = glob.glob("mock_docs/*.txt")
    
    if not document_files:
        st.warning("No documents found in `mock_docs/`. Please generate them first by running `python src/generate_documents.py`.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Score a Single Document")
            selected_doc = st.selectbox("Choose a document to score:", [os.path.basename(f) for f in document_files])
            
            if st.button("Score Selected Document"):
                if selected_doc:
                    doc_path = os.path.join("mock_docs", selected_doc)
                    with st.spinner(f"Scoring {selected_doc}..."):
                        score_result = score_lead(doc_path, scorer_chain)
                        if score_result and 'score' in score_result:
                            st.metric(label="Lead Score", value=f"{score_result['score']}/100")
                            st.info(f"**Rationale:** {score_result['rationale']}")
                        else:
                            st.error("Could not retrieve a valid score.")

        with col2:
            st.subheader("Score All Documents")
            if st.button("Score All Documents"):
                with st.spinner("Scoring all lead documents... This may take a moment."):
                    all_scores = [score_lead(doc, scorer_chain) for doc in document_files]
                    # Filter out unsuccessful scores and sort
                    successful_scores = [s for s in all_scores if s and 'score' in s]
                    sorted_scores = sorted(successful_scores, key=lambda x: x['score'], reverse=True)
                    
                    st.dataframe(sorted_scores, use_container_width=True)