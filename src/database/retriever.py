from langchain_chroma import Chroma
from src.database.vector_store import get_embedding_function
from src.config import DB_DIR

def get_retriever(k=3):
    """
    Creates a retriever object that searches the ChromaDB.
    
    Args:
        k (int): The number of chunks to return (Assignment asks for 3-5).
        
    Returns:
        VectorStoreRetriever: A callable object that performs the search.
    """
    embedding_fn = get_embedding_function()
    
    # Architecture Decision: Load from Disk (Persistence)
    # Reasoning: We don't create a new DB here; we just connect to the 
    # folder where 'vector_store.py' saved the data. 
    # This allows the retrieval to happen instantly without re-processing PDFs.
    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_fn
    )
    
    # Architecture Decision: Similarity Search with k=3
    # Reasoning: We use standard cosine similarity. 
    # k=3 retrieves roughly 750-1000 words of context. This is enough for 
    # the LLM to understand the topic without getting confused by irrelevant details.
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    return retriever