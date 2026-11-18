from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = str(PROJECT_ROOT / "data" / "raw")


def load_documents(data_path: str = DATA_PATH):
    """Loads all PDF documents from the specified directory."""
    loader = DirectoryLoader(
        path=data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} document pages.")
