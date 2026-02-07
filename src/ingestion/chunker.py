from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_documents(documents):
    """
    Splits a list of documents into smaller chunks based on the config settings.
    
    Args:
        documents (List[Document]): The raw documents loaded from the PDF.
        
    Returns:
        List[Document]: A list of smaller chunked documents with metadata preserved.
    """
    if not documents:
        print("No documents to chunk.")
        return []

    print(f"Chunking {len(documents)} documents with size={CHUNK_SIZE} and overlap={CHUNK_OVERLAP}...")

    # We use RecursiveCharacterTextSplitter to maintain semantic context.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""], 
        length_function=len,
        add_start_index=True 
    )

    chunks = text_splitter.split_documents(documents)
    
    print(f"Split into {len(chunks)} chunks.")
    return chunks