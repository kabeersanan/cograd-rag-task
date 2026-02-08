import os
import shutil
from langchain_chroma import Chroma
# Using Local Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import DB_DIR

def get_embedding_function():
    # Using Local MPNet (512 tokens) as discussed
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )

def create_vector_db(chunks):
    if not chunks: return
    if os.path.exists(DB_DIR): shutil.rmtree(DB_DIR)
    
    embedding_fn = get_embedding_function()
    vector_store = Chroma.from_documents(chunks, embedding_fn, persist_directory=DB_DIR)
    return vector_store

def load_vector_db():
    if not os.path.exists(DB_DIR): raise FileNotFoundError(f"No DB at {DB_DIR}")
    embedding_fn = get_embedding_function()
    return Chroma(persist_directory=DB_DIR, embedding_function=embedding_fn)