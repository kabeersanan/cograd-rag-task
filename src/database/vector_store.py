import os
import shutil
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import DB_DIR

def get_embedding_function():
    """
    MODEL: sentence-transformers/all-mpnet-base-v2
    CONTEXT WINDOW: 512 Tokens
    
    REASONING:
    The user has chunks up to 500 tokens.
    - MiniLM (256) would truncate data.
    - MPNet (512) fits the data perfectly without truncation.
    """
    print("Initialize Local Embeddings: all-mpnet-base-v2 (512 token limit)...")
    
    # model_kwargs={'device': 'cpu'} ensures it runs on any laptop.
    # encode_kwargs={'normalize_embeddings': True} improves cosine similarity search.
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def create_vector_db(chunks):
    """
    Generates embeddings locally and saves to disk.
    """
    if not chunks:
        print("No chunks to process.")
        return

    # 1. Reset Database (Critical to avoid shape mismatches)
    if os.path.exists(DB_DIR):
        print(f"Cleaning old database at {DB_DIR}...")
        shutil.rmtree(DB_DIR)

    print(f"Generating embeddings for {len(chunks)} chunks...")
    
    embedding_fn = get_embedding_function()
    
    # 2. Create and Persist
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=DB_DIR
    )
    
    print(f"Vector Database successfully created at {DB_DIR}")
    return vector_store

def load_vector_db():
    """
    Loads the existing local vector database.
    """
    if not os.path.exists(DB_DIR):
        raise FileNotFoundError(f"No database found at {DB_DIR}. Run the ingestion pipeline first.")

    embedding_fn = get_embedding_function()
    
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_fn
    )