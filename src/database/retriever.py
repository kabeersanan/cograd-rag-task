from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from src.database.vector_store import get_embedding_function
from src.config import DB_DIR

def get_retriever(k=3):
    """
    Creates a HYBRID retriever that combines Semantic Search (Vector) 
    with Keyword Search (BM25).
    
    Args:
        k (int): Number of chunks to retrieve (Assignment asks for 3-5).
        
    Returns:
        EnsembleRetriever: A robust retriever combining two strategies.
    """
    embedding_fn = get_embedding_function()
    
    # 1. Initialize Vector Store (Semantic Search)
    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_fn
    )
    
    # Reconstruction for Hybrid Search
    # To use BM25 (Keyword Search), we need the raw text of the chunks.
    # Instead of re-reading PDFs, we pull the text directly from our Vector DB.
    try:
        # Fetch all stored documents from Chroma
        data = vector_store.get() 
        texts = data['documents']
        metadatas = data['metadatas']
        
        if not texts:
            print(" Warning: Vector DB is empty. utilizing fallback.")
            return vector_store.as_retriever(search_kwargs={"k": k})

        # 2. Initialize BM25 Retriever (Keyword Search)
        # This catches specific terms like "Fe2O3" or "displacement" that vectors might miss.
        bm25_retriever = BM25Retriever.from_texts(texts, metadatas=metadatas)
        bm25_retriever.k = k

        # 3. Initialize Standard Vector Retriever
        chroma_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

        # 4. Create Ensemble (Hybrid) Retriever
        # Weighting: 50% Semantic (Concepts) + 50% Keyword (Precision)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.5, 0.5]
        )
        
        print(f"Hybrid Retriever initialized (BM25 + Chroma) with k={k}")
        return ensemble_retriever

    except Exception as e:
        print(f" Error initializing Hybrid Search: {e}")
        print("Falling back to standard Vector Search.")
        return vector_store.as_retriever(search_kwargs={"k": k})