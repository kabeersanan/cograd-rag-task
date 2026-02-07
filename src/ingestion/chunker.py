from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_documents(documents):
    """
    Splits documents into smaller chunks using an 'Intelligent' strategy.
    
    Strategy:
    1. It looks for Paragraph breaks (\n\n) first.
    2. Then it looks for Line breaks (\n) to keep Headings intact.
    3. Then it looks for Sentences (. ) to keep ideas intact.
    """
    if not documents:
        print("No documents to chunk.")
        return []

    print(f"Chunking {len(documents)} documents intelligently...")

    # We use RecursiveCharacterTextSplitter to maintain semantic context.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
        # INTELLIGENT CHUNKING UPGRADE:
        # The order matters! It tries to split by the first separator, then the second, etc.
        separators=[
            "\n\n",  # 1. Keep Paragraphs together (Best for topics)
            "\n",    # 2. Keep Headings/Lists together
            ". ",    # 3. Keep Sentences together (Critical addition!)
            " ",     # 4. Split by words
            ""       # 5. Split by characters (Worst case)
        ],
        is_separator_regex=False
    )

    chunks = text_splitter.split_documents(documents)
    
    print(f"Split into {len(chunks)} smart chunks.")
    return chunks