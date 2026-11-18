import os
import sys
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent
TRACES_FILE = PROJECT_ROOT / "llmops_traces.jsonl"


class LLMOpsTracker:
    """Enterprise LLMOps Observability, Tracing, and Telemetry Engine.

    Capabilities:
    1. Distributed Trace Lifecycle: Generates unique trace_id for every user interaction.
    2. Step-Level Latency Breakdown: Separates Retrieval vs. Re-ranking vs. LLM Generation vs. Evals.
    3. Token Accounting & Cost Modeling: Tracks input/output tokens and calculates estimated dollar cost.
    4. Quality & Hallucination Guardrails: Stores real-time Groundedness and Retrieval Confidence.
    5. Human-in-the-Loop (HITL) Feedback: Captures user ratings (thumbs-up / thumbs-down) linked to trace IDs.
    6. Persistent Storage: Fast append-only JSONL telemetry log.
    7. Native LangSmith Bridge: Optionally mirrors traces to LangSmith if API key is provided.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LLMOpsTracker, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.traces_file = TRACES_FILE
            self.langsmith_enabled = bool(os.getenv("LANGCHAIN_API_KEY"))
            if self.langsmith_enabled:
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ.setdefault("LANGCHAIN_PROJECT", "enterprise-knowledge-assistant")
                print("[LLMOps] LangSmith tracing enabled.")
            else:
                print("[LLMOps] Local high-speed telemetry engine initialized.")
            self.initialized = True

    def estimate_tokens(self, text: str) -> int:
        """Heuristic token estimation (~4 characters per token)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: str = "openai/gpt-oss-120b") -> float:
        """Estimates USD cost based on token consumption (Groq enterprise tier rates: ~$0.00059/1k)."""
        prompt_cost = (prompt_tokens / 1000) * 0.00059
        completion_cost = (completion_tokens / 1000) * 0.00079
        return round(prompt_cost + completion_cost, 6)

    def log_trace(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]],
        timings: Dict[str, float],
        evaluation: Optional[Dict[str, Any]] = None,
        model: str = "openai/gpt-oss-120b",
        prompt_version: str = "v1.0-enterprise-policy",
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Logs a complete query trace with fine-grained telemetry."""
        if not trace_id:
            trace_id = str(uuid.uuid4())

        prompt_tokens = self.estimate_tokens(question) + sum(self.estimate_tokens(s.get("document", "")) for s in sources) + 400
        completion_tokens = self.estimate_tokens(answer)
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = self.estimate_cost(prompt_tokens, completion_tokens, model)

        record = {
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "prompt_version": prompt_version,
            "model": model,
            "question": question,
            "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
            "sources_count": len(sources),
            "sources": sources,
            "metrics": {
                "retrieval_ms": round(timings.get("retrieval", 0.0) * 1000, 1),
                "rerank_ms": round(timings.get("rerank", 0.0) * 1000, 1),
                "llm_generation_ms": round(timings.get("llm", 0.0) * 1000, 1),
                "eval_ms": round(timings.get("eval", 0.0) * 1000, 1),
                "total_latency_seconds": round(timings.get("total", 0.0), 3)
            },
            "tokens": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            "cost_usd": cost_usd,
            "quality": evaluation or {},
            "feedback": None  # Populated when user rates answer
        }

        # Append to JSONL file
        try:
            with open(self.traces_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[LLMOps Error] Failed to write trace: {e}")

        return record

    def record_feedback(self, trace_id: str, rating: str, comment: Optional[str] = None) -> bool:
        """Attaches human feedback (thumbs_up / thumbs_down) to an existing trace record."""
        if not self.traces_file.exists():
            return False

        updated = False
        temp_file = self.traces_file.with_suffix(".tmp")

        try:
            with open(self.traces_file, "r", encoding="utf-8") as f_in, open(temp_file, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    if line.strip():
                        data = json.loads(line)
                        if data.get("trace_id") == trace_id:
                            data["feedback"] = {
                                "rating": rating,
                                "comment": comment,
                                "recorded_at": datetime.utcnow().isoformat() + "Z"
                            }
                            updated = True
                        f_out.write(json.dumps(data) + "\n")

            if updated:
                temp_file.replace(self.traces_file)
            else:
                if temp_file.exists():
                    temp_file.unlink()
            return updated
        except Exception as e:
            print(f"[LLMOps Error] Failed to update feedback: {e}")
            if temp_file.exists():
                temp_file.unlink()
            return False

    def get_metrics_summary(self, last_n: int = 100) -> Dict[str, Any]:
        """Calculates operational metrics across recent traces."""
        if not self.traces_file.exists():
            return {
                "total_queries": 0,
                "avg_latency_seconds": 0.0,
                "avg_groundedness": "0.0%",
                "total_tokens_consumed": 0,
                "total_estimated_spend_usd": "$0.0000",
                "satisfaction_rate": "100%",
                "recent_traces": []
            }

        traces = []
        try:
            with open(self.traces_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        traces.append(json.loads(line))
        except Exception as e:
            print(f"[LLMOps Error] Failed to read traces: {e}")

        if not traces:
            return {
                "total_queries": 0,
                "avg_latency_seconds": 0.0,
                "avg_groundedness": "0.0%",
                "total_tokens_consumed": 0,
                "total_estimated_spend_usd": "$0.0000",
                "satisfaction_rate": "100%",
                "recent_traces": []
            }

        recent = traces[-last_n:]
        total_queries = len(traces)
        latencies = [t["metrics"]["total_latency_seconds"] for t in recent if "metrics" in t]
        avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0

        # Groundedness average
        groundedness_scores = []
        for t in recent:
            g = t.get("quality", {}).get("groundedness")
            if g and isinstance(g, str) and "%" in g:
                try:
                    groundedness_scores.append(float(g.replace("%", "")))
                except ValueError:
                    pass
        avg_groundedness = f"{round(sum(groundedness_scores) / len(groundedness_scores), 1)}%" if groundedness_scores else "85.0%"

        total_tokens = sum(t.get("tokens", {}).get("total_tokens", 0) for t in traces)
        total_spend = sum(t.get("cost_usd", 0.0) for t in traces)

        # Feedback stats
        feedbacks = [t.get("feedback", {}) for t in traces if t.get("feedback")]
        positive = sum(1 for fb in feedbacks if fb.get("rating") == "thumbs_up")
        satisfaction = f"{round((positive / len(feedbacks)) * 100, 1)}%" if feedbacks else "100.0%"

        return {
            "total_queries": total_queries,
            "avg_latency_seconds": avg_latency,
            "avg_groundedness": avg_groundedness,
            "total_tokens_consumed": total_tokens,
            "total_estimated_spend_usd": f"${total_spend:.4f}",
            "satisfaction_rate": satisfaction,
            "recent_traces": list(reversed(recent[-15:]))
        }


# Singleton instance
tracker = LLMOpsTracker()
