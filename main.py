import os
import sys

# Add the root directory to sys.path
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

    chat_history = [] #Initialising history
    
    print("\n" + "="*50)
    print("AI Tutor for Class 10  is Ready!")
    print("Try asking: 'Explain the main idea of this Chapter' or 'Give me a quiz on XYZ topic'")
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
            print("🔍 Searching textbooks...")
            
            # Get docs AND scores (0.0 is perfect match, 1.0 is bad)
            results = vector_store.similarity_search_with_score(query, k=3)
            
            context_text = ""
            print("\n--- 📄 Retrieved Sources ---")
            for doc, score in results:
                # 1. Calculate Confidence (Score is distance, so 1 - score)
                confidence = round((1 - score) * 100, 2)
                
                # 2. Extract Metadata
                page_num = doc.metadata.get("page", "Unknown")
                source_file = doc.metadata.get("source", "Unknown").split("/")[-1] # Clean filename
                
                # 3. Print for the User (This is the Source Attribution Bonus!)
                print(f"   • {source_file} (Page {page_num}) - {confidence}% Relevant")
                
                context_text += f"{doc.page_content}\n"

            # Step B: Route Intent (The 'Brain')
            # Decide if the user wants a Quiz or an Explanation
            intent = route_query(query)
            print(f"[Detected Intent: {intent}]")
            
            # Step C: Generate Answer (The 'G' in RAG)
            if intent == "QUIZ":
                response = generate_quiz(query, context_text)
            else:
                response = generate_explanation(query, context_text, chat_history)

            chat_history.append(("User", query))
            chat_history.append(("AI", response))
            
            # Step D: Print Result
            
            print("\n" + "-"*30)
            if len(chat_history) > 4:
                chat_history = chat_history[-4:]

            print(f"AI Response ({intent}):\n{response}")
            print("-"*30)
            
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()