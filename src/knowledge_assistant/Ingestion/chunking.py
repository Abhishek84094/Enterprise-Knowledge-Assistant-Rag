import sys
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ensure project root is in sys.path for clean imports
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import document loader from document_loader.py
try:
    from document_loader import load_documents
except ImportError:
    from src.knowledge_assistant.Ingestion.document_loader import load_documents


def chunk_documents(docs, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Splits loaded documents into smaller overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(docs)


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)

    print(f"Total chunks created: {len(chunks)}")
    if chunks:
        print("\n--- Sample Chunk Preview ---")
        print(f"Source: {chunks[0].metadata.get('source')}")
        print(f"Page: {chunks[0].metadata.get('page')}")
        print(f"Content:\n{chunks[0].page_content[:300]}...")
        print(chunks[:1])