import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Setup paths and environment
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Load environment variables from .env
dotenv_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Default paths
FAISS_INDEX_DIR = str(PROJECT_ROOT / "faiss_index")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Import ingestion modules
try:
    from document_loader import load_documents
    from chunking import chunk_documents
except ImportError:
    from src.knowledge_assistant.Ingestion.document_loader import load_documents
    from src.knowledge_assistant.Ingestion.chunking import chunk_documents


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Initializes and returns the Hugging Face embedding model."""
    print(f"Loading embedding model: {model_name}...")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def build_and_save_vector_store(chunks, save_path: str = FAISS_INDEX_DIR, embedding_model=None):
    """Generates embeddings for chunks and persists the FAISS vector database to disk."""
    if embedding_model is None:
        embedding_model = get_embedding_model()

    print(f"Generating embeddings and building FAISS index for {len(chunks)} chunks...")
    vector_store = FAISS.from_documents(chunks, embedding_model)

    print(f"Saving FAISS index to: {save_path}")
    vector_store.save_local(save_path)
    print("Vector database saved successfully!")
    return vector_store


def load_vector_store(save_path: str = FAISS_INDEX_DIR, embedding_model=None):
    """Loads an existing FAISS vector database from disk."""
    if embedding_model is None:
        embedding_model = get_embedding_model()

    print(f"Loading FAISS index from: {save_path}...")
    return FAISS.load_local(
        folder_path=save_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )


if __name__ == "__main__":
    print("=== Step 1: Loading Documents ===")
    docs = load_documents()
    print(f"Loaded {len(docs)} document pages.")

    print("\n=== Step 2: Chunking Documents ===")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("\n=== Step 3: Generating Embeddings & Storing in FAISS ===")
    embedding_model = get_embedding_model()
    vector_store = build_and_save_vector_store(chunks, FAISS_INDEX_DIR, embedding_model)

    print("\n=== Step 4: Verification - Similarity Search ===")
    test_query = "What is the policy for remote work and expense reimbursement?"
    print(f"Query: '{test_query}'")
    results = vector_store.similarity_search_with_score(test_query, k=2)

    for i, (doc, score) in enumerate(results, 1):
        print(f"\nResult {i} (Distance Score: {score:.4f}):")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Page: {doc.metadata.get('page')}")
        print(f"Content:\n{doc.page_content[:250]}...")

