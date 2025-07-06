import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

def load_and_chunk_documents(directory_path="mock_docs"):
    """
    Loads documents from a directory and splits them into chunks.
    """
    print(f"Loading documents from {directory_path}...")
    # Using TextLoader to specify UTF-8 encoding
    loader = DirectoryLoader(
        directory_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},
        show_progress=True
    )
    documents = loader.load()

    if not documents:
        print("No documents found. Please generate documents first.")
        return []

    print(f"Loaded {len(documents)} documents.")

    # Using RecursiveCharacterTextSplitter to split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,  # ~450 words
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    
    return chunks

if __name__ == "__main__":
    chunks = load_and_chunk_documents()
    if chunks:
        print("\n--- Example Chunk ---")
        print(chunks[0].page_content)
        print("--------------------")
        print(f"Metadata: {chunks[0].metadata}") 