import os
import sys

# Add the root directory to sys.path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.pdf_loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.database.vector_store import create_vector_db, load_vector_db
from src.database.retriever import get_retriever
from src.agents.router import route_query
from src.agents.concept_agent import generate_explanation
from src.agents.quiz_agent import generate_quiz
from src.config import DB_DIR

def initialize_system():
    # This function checks if we already have a database.
    # If yes, it loads it. If no, it creates it from the PDF.
    
    print("--- System Initialization ---")
    
    # Check if the database folder exists and is not empty
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print("Vector Database found. Loading existing memory...")
        return load_vector_db()
    else:
        print("No Vector Database found. Starting Ingestion Pipeline...")
        
        # Step 1: Load
        documents = load_documents()
        if not documents:
            print("CRITICAL ERROR: No PDFs found in data/raw/. Please add a file.")
            sys.exit(1)
        
        # Step 2: Chunk
        chunks = chunk_documents(documents)
        
        # Step 3: Store
        vector_store = create_vector_db(chunks)
        return vector_store

def main():
    # 1. Start the System
    # This might take a few seconds if it needs to process the PDF
    vector_store = initialize_system()
    
    # 2. Create the Retriever
    # This tool will fetch the relevant text for us
    retriever = get_retriever()
    
    print("\n" + "="*50)
    print("AI Tutor for Class 10 Science is Ready!")
    print("Try asking: 'Explain displacement reactions' or 'Give me a quiz on acids'")
    print("Type 'exit' to quit.")
    print("="*50 + "\n")

    # 3. The Main Interaction Loop
    while True:
        # Get user input
        query = input("\nYou: ")
        
        # Check for exit command
        if query.lower() in ["exit", "quit", "bye"]:
            print("Goodbye! Happy Studying! ")
            break

        # Skip empty inputs
        if not query.strip():
            continue

        try:
            print("... Thinking ...")
            
            # Step A: Retrieve Context (The 'R' in RAG)
            # We find the 3 most relevant paragraphs from the book
            docs = retriever.invoke(query)
            
            # Combine them into a single string of text
            context_text = "\n\n".join([d.page_content for d in docs])
            
            if not context_text:
                print("I could not find any relevant information in the provided chapter.")
                continue

            # Step B: Route Intent (The 'Brain')
            # Decide if the user wants a Quiz or an Explanation
            intent = route_query(query)
            print(f"[Detected Intent: {intent}]")
            
            # Step C: Generate Answer (The 'G' in RAG)
            if intent == "QUIZ":
                response = generate_quiz(query, context_text)
            else:
                response = generate_explanation(query, context_text)
            
            # Step D: Print Result
            print("\n" + "-"*30)
            print(f"AI Response ({intent}):\n")
            print(response)
            print("-"*30)
            
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()