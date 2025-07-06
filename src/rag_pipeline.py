import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def load_rag_pipeline(api_key: str):
    """
    Loads the entire RAG pipeline, using the OpenRouter API.
    """
    vector_store_path = "faiss_index"
    if not os.path.exists(vector_store_path):
        raise FileNotFoundError(
            f"Vector store not found at {vector_store_path}. "
            "Please run `create_vector_store.py` first."
        )

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Use OpenRouter's free DeepSeek model
    llm = ChatOpenAI(
        model="deepseek/deepseek-chat:free",
        api_key=api_key,  # type: ignore
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_tokens=1024
    )

    # This prompt is a best practice for general instruction-following models
    template = """
    You are an expert financial analyst. Your goal is to provide a detailed, professional, and well-structured answer based *only* on the context provided.
    
    Analyze the following context, synthesize the information, and answer the user's question thoroughly. If the context does not contain the answer, state that clearly. Do not make up information.

    CONTEXT:
    {context}

    QUESTION:
    {input}
    """
    prompt = PromptTemplate.from_template(template)

    # Create the RAG chain
    chain = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(llm, prompt)
    )
    return chain

def ask_question(query: str, chain):
    """Invokes the RAG chain with the user's query."""
    if not query:
        return {"answer": "Please enter a query.", "context": []}
    try:
        result = chain.invoke({"input": query})
        return result
    except Exception as e:
        return {"answer": f"An error occurred: {e}", "context": []}

if __name__ == "__main__":
    try:
        api_key = input("Enter your OpenRouter API key: ")
        rag_chain = load_rag_pipeline(api_key)
        print("RAG pipeline loaded successfully. You can now ask questions.")
        
        # Example query
        example_query = "What is the typical duration for a small business loan?"
        print(f"\n--- Asking example question: '{example_query}' ---")
        ask_question(example_query, rag_chain)

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}") 