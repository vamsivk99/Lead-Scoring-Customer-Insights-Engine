import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import SecretStr
from src.preprocess import load_and_chunk_documents

def create_vector_store(output_path="faiss_index"):
    """
    Creates a FAISS vector store from the documents and saves it to disk.
    """
    load_dotenv()

    # 1. Load and chunk documents
    chunks = load_and_chunk_documents()
    if not chunks:
        return

    # 2. Initialize embeddings model
    try:
        # Using a local, open-source embedding model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Error initializing HuggingFace Embeddings: {e}")
        return

    # 3. Create FAISS vector store
    print("Creating FAISS vector store...")
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        print(f"Error creating FAISS vector store: {e}")
        # This can happen due to API errors, e.g., invalid key
        return
        
    # 4. Save the vector store
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    vector_store.save_local(output_path)
    print(f"Vector store created and saved to '{output_path}'.")

if __name__ == "__main__":
    create_vector_store() 