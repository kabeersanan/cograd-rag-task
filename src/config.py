import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# --- API KEYS ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- FILE PATHS ---
# We use os.path.join to make sure it works on both Windows and Mac
DATA_DIR = os.path.join("data", "raw")
DB_DIR = os.path.join("data", "vector_store")

# --- RAG SETTINGS ---
# Chunk Size 1000: Good balance. Large enough to capture full context (approx 2-3 paragraphs).
# Overlap 200: Ensures we don't cut a sentence in half at the edge of a chunk.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- AI MODELS ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Using Gemini Flash because it is fast, cheap, and has a large context window
LLM_MODEL_NAME = "llama-3.3-70b-versatile"