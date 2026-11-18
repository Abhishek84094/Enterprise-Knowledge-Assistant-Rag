import os
import sys
import time
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Enforce offline HuggingFace loading to prevent slow network ping delays
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import RAG Chain
try:
    from rag_chain import EnterpriseRAGChain
except ImportError:
    from src.knowledge_assistant.rag_chain import EnterpriseRAGChain

# Initialize FastAPI app
app = FastAPI(
    title="Enterprise Knowledge Assistant API",
    description="FastAPI backend for Enterprise RAG with FAISS, Cross-Encoder Re-ranking, and Groq LLM",
    version="1.0.0"
)

# Enable CORS for React frontend (Vite defaults to port 5173, CRA to 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG Chain singleton
rag_instance: Optional[EnterpriseRAGChain] = None


def get_rag():
    """Lazily initializes and caches the EnterpriseRAGChain instance."""
    global rag_instance
    if rag_instance is None:
        print("[API] Initializing EnterpriseRAGChain singleton...")
        rag_instance = EnterpriseRAGChain()
    return rag_instance


@app.on_event("startup")
def prewarm_models():
    """Pre-warms the RAG chain on server startup so users experience zero cold-start delay."""
    try:
        print("[STARTUP] Pre-warming FAISS and Cross-Encoder in memory...")
        get_rag()
        print("[STARTUP] Enterprise RAG ready for instant queries!")
    except Exception as e:
        print(f"[STARTUP WARNING] Pre-warm failed: {e}")


# Pydantic Schemas
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, description="Employee question regarding company policies", example="What is the meal reimbursement cap?")


class SourceItem(BaseModel):
    document: str
    page: Optional[int | str] = None
    relevance_score: float


class RealtimeEval(BaseModel):
    retrieval_confidence: str
    groundedness: str
    answer_relevance: str
    hallucination_risk: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]
    latency_seconds: float
    evaluation: Optional[RealtimeEval] = None
    trace_id: Optional[str] = None
    tokens: Optional[dict] = None
    cost_usd: Optional[float] = None
    llmops_metrics: Optional[dict] = None


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: str = Field(..., description="thumbs_up or thumbs_down")
    comment: Optional[str] = None


class DocumentInfo(BaseModel):
    filename: str
    size_kb: float


class HealthResponse(BaseModel):
    status: str
    rag_ready: bool
    groq_model: str
    faiss_index_exists: bool


# API Routes
@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint to verify backend status and vector store readiness."""
    faiss_dir = PROJECT_ROOT / "faiss_index"
    return HealthResponse(
        status="healthy",
        rag_ready=rag_instance is not None,
        groq_model=os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b"),
        faiss_index_exists=faiss_dir.exists()
    )


@app.get("/api/documents", response_model=List[DocumentInfo])
def list_documents():
    """Returns the list of ingested enterprise policy documents available in the knowledge base."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    if not raw_dir.exists():
        return []

    docs = []
    for file in sorted(raw_dir.glob("*.pdf")):
        docs.append(DocumentInfo(
            filename=file.name,
            size_kb=round(file.stat().st_size / 1024, 2)
        ))
    return docs


@app.get("/api/llmops/metrics")
def get_llmops_metrics():
    """Returns operational LLMOps metrics: total queries, latency, tokens, cost, satisfaction rate, and recent traces."""
    try:
        from llmops import tracker
    except ImportError:
        from src.knowledge_assistant.llmops import tracker
    return tracker.get_metrics_summary()


@app.post("/api/llmops/feedback")
def submit_llmops_feedback(payload: FeedbackRequest):
    """Records human-in-the-loop (thumbs-up / thumbs-down) feedback linked to a query trace ID."""
    try:
        from llmops import tracker
    except ImportError:
        from src.knowledge_assistant.llmops import tracker
    success = tracker.record_feedback(payload.trace_id, payload.rating, payload.comment)
    return {"success": success, "trace_id": payload.trace_id}


@app.post("/api/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest):
    """Processes an employee question through the LCEL RAG pipeline with Re-ranking and Online Evals."""
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        rag = get_rag()
        start_time = time.time()
        result = rag.ask(payload.question)
        elapsed = round(time.time() - start_time, 2)

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=[
                SourceItem(
                    document=s["document"],
                    page=s["page"],
                    relevance_score=s["relevance_score"]
                )
                for s in result.get("sources", [])
            ],
            latency_seconds=elapsed,
            evaluation=result.get("evaluation"),
            trace_id=result.get("trace_id"),
            tokens=result.get("tokens"),
            cost_usd=result.get("cost_usd"),
            llmops_metrics=result.get("llmops_metrics")
        )
    except Exception as e:
        print(f"[API ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG execution failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

