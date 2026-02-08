import os
import sys
import shutil

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.pdf_loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.database.vector_store import create_vector_db
from src.database.retriever import get_retriever
from src.agents.router import route_query
from src.agents.concept_agent import generate_explanation
from src.agents.quiz_agent import generate_quiz
from src.config import DB_DIR

def ensure_knowledge_base():
    """
    Checks if the Vector DB exists. If not, builds it from scratch.
    """
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print(f"Knowledge Base found in {DB_DIR}. Skipping ingestion.")
        return

    print("No Knowledge Base found. Starting Ingestion Pipeline...")
    
    # Step 1: Load
    documents = load_documents()
    if not documents:
        print("CRITICAL ERROR: No PDFs found in data/raw/. Please add a file.")
        sys.exit(1)
    
    # Step 2: Intelligent Chunking (Token-based + Metadata)
    chunks = chunk_documents(documents)
    
    # Step 3: Store in Vector DB
    create_vector_db(chunks)
    print("Ingestion Complete. Database built.")

def format_docs_for_agent(docs):
    """
    Prepares retrieved documents for the LLM.
    """
    return "\n\n".join([f"Content: {d.page_content}\nSource: Page {d.metadata.get('page', 'Unknown')}" for d in docs])

def main():
    # 1. System Check
    ensure_knowledge_base()
    
    # 2. Initialize Hybrid Retriever (Bonus Feature)
    # This combines Keyword Search (BM25) and Semantic Search (Chroma)
    retriever = get_retriever(k=4) 
    
    chat_history = [] 
    
    print("\n" + "="*60)
    print("🎓 AI Tutor for Class 10- powered by Hybrid RAG")
    print("   - Intelligent Chunking: ON")
    print("   - Hybrid Search (BM25 + Vector): ON")
    print("   - Source Attribution: ON")
    print("="*60 + "\n")

    # 3. Main Loop
    while True:
        query = input("\nStudent: ")
        
        if query.lower() in ["exit", "quit", "bye", "stop"]:
            print("Goodbye! Happy Studying!")
            break

        if not query.strip():
            continue

        try:
            print("Thinking...")

            # --- Step A: Hybrid Retrieval ---
            # We use invoke() because EnsembleRetriever manages the logic
            retrieved_docs = retriever.invoke(query)
            
            # Display Sources (Bonus: Source Attribution)
            print("\n Retrieved Sources (Hybrid Search):")
            for i, doc in enumerate(retrieved_docs[:3]):
                page = doc.metadata.get("page", "Unknown")
                topic = doc.metadata.get("topic", "General")
                print(f"   {i+1}. [Page {page}] Topic: {topic}...")

            context_text = format_docs_for_agent(retrieved_docs)

            # --- Step B: Intent Routing  ---
            intent = route_query(query).strip().upper()
            
            if "QUIZ" in intent:
                print(f"\n Generating Quiz on: {query}")
                response = generate_quiz(query, context_text)
            
            elif "CHAT" in intent:
                # Simple conversational fallback
                response = "Hello! I am your AI Tutor. Ask me anything about your Class 10 chapter."
            
            else: # Default to EXPLAIN
                print(f"\n Generating Explanation...")
                response = generate_explanation(query, context_text, chat_history)

            # --- Step C: Update History & Display ---
            chat_history.append((query, response))
            if len(chat_history) > 3:
                chat_history.pop(0)

            print("\n" + "="*50)
            print(f" AI Tutor ({intent}):")
            
            # Check if response is a list (Quiz) or string (Explanation)
            if isinstance(response, list):
                for i, q in enumerate(response):
                    print(f"\nQ{i+1}: {q['question']}")
                    for option in q['options']:
                        print(f"   {option}")
                    
                    # Hidden Answer Key (Optional - maybe show after user hits enter?)
                    print(f"   [Answer: {q['answer']} | Reason: {q['explanation']}]")
            else:
                # It's a text explanation
                print(f"\n{response}")
                
            print("="*50)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()