import os
import shutil
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import DB_DIR, EMBEDDING_MODEL_NAME, GOOGLE_API_KEY

def get_embedding_function():
    """
    Returns the configured Google GenAI embedding function.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is missing. Please check your .env file.")

    # Architecture Decision: Use Google's dedicated embedding model
    # Reasoning: 'models/text-embedding-004' is optimized for semantic retrieval
    # and has a higher dimensionality (better accuracy) than older models.
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY
    )

def create_vector_db(chunks):
    """
    Takes text chunks, generates embeddings, and saves them to disk.
    
    Args:
        chunks (List[Document]): The chunked text from the PDF.
    """
    if not chunks:
        print("No chunks to process.")
        return

    # 1. Clear existing DB (Optional but recommended for development)
    # Reasoning: If we run this script twice, we don't want duplicate data.
    # We remove the old DB folder and start fresh.
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    print(f"Generating embeddings for {len(chunks)} chunks...")
    
    embedding_fn = get_embedding_function()
    
    # 2. Create and Persist ChromaDB
    # Architecture Decision: Local Persistence
    # Reasoning: We use 'persist_directory' so the data is saved to the hard drive.
    # This means you don't have to pay for embeddings every time you restart the app.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=DB_DIR
    )
    
    print(f"Vector Database successfully created at {DB_DIR}")
    return vector_store

def load_vector_db():
    """
    Loads the existing vector database from disk without re-ingesting data.
    """
    if not os.path.exists(DB_DIR):
        raise FileNotFoundError(f"No database found at {DB_DIR}. Run the ingestion pipeline first.")

    embedding_fn = get_embedding_function()
    
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_fn
    )