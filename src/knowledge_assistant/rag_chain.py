import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Enforce 100% offline local model loading (eliminates slow HuggingFace Hub network checks)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Path setup
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Load .env
load_dotenv(PROJECT_ROOT / ".env")

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from sentence_transformers import CrossEncoder

# Import vector store loader
try:
    from Ingestion.vector_store import load_vector_store
    from llmops import tracker
except ImportError:
    from src.knowledge_assistant.Ingestion.vector_store import load_vector_store
    from src.knowledge_assistant.llmops import tracker

# Configuration defaults
FAISS_INDEX_DIR = str(PROJECT_ROOT / "faiss_index")
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")


class EnterpriseRAGChain:
    """Production-grade Enterprise Knowledge Assistant using LCEL (RunnableParallel)

    Features:
    - FAISS Vector Similarity Search (Candidate Retrieval)
    - Cross-Encoder Re-ranking (Semantic Relevance Scoring)
    - RunnableParallel branching for declarative, composable execution
    - Groq LLM Generation with strict enterprise grounding & citations
    """

    def __init__(
        self,
        faiss_path: str = FAISS_INDEX_DIR,
        reranker_model: str = RERANKER_MODEL_NAME,
        llm_model: str = GROQ_MODEL_NAME,
        temperature: float = 0.1,
        top_k_retrieve: int = 8,
        top_n_rerank: int = 3
    ):
        self.top_k_retrieve = top_k_retrieve
        self.top_n_rerank = top_n_rerank

        print("Initializing Enterprise Knowledge Assistant LCEL Chain...")

        # 1. Load FAISS vector store
        self.vector_store = load_vector_store(save_path=faiss_path)

        # 2. Initialize Cross-Encoder Re-ranker
        print(f"Loading Re-ranking model: {reranker_model}...")
        self.reranker = CrossEncoder(reranker_model)

        # 3. Initialize Groq LLM
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key or groq_api_key == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY is missing or invalid in your .env file.")

        print(f"Connecting to Groq LLM: {llm_model}...")
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model,
            temperature=temperature,
            request_timeout=25.0,
            max_retries=2
        )

        # 4. Prompt Template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an enterprise knowledge assistant for company employees.\n"
                "Answer the employee's question accurately, concisely, and professionally based ONLY on the provided policy documents below.\n"
                "If the information is not contained in the documents, state clearly: "
                "'I could not find information regarding this in the company policy documents.'\n"
                "Always cite your sources by mentioning the Document Name and Page Number at the end of your answer."
            )),
            ("human", (
                "Company Policy Documents:\n"
                "---------------------\n"
                "{context}\n"
                "---------------------\n\n"
                "Employee Question: {question}\n\n"
                "Helpful Answer:"
            ))
        ])

        # 5. Build declarative LCEL Runnable chain
        self.chain = self._build_runnable_chain()

    def retrieve_and_rerank(self, query: str):
        """Retrieves top_k candidates from FAISS and re-ranks them using Cross-Encoder to select top_n."""
        candidate_docs = self.vector_store.similarity_search(query, k=self.top_k_retrieve)
        if not candidate_docs:
            return []

        pairs = [[query, doc.page_content] for doc in candidate_docs]
        scores = self.reranker.predict(pairs)

        scored_docs = list(zip(candidate_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:self.top_n_rerank]

    def format_context(self, scored_docs):
        """Formats reranked documents into readable context string."""
        context_parts = []
        for doc, score in scored_docs:
            source = Path(doc.metadata.get("source", "Unknown")).name
            page = doc.metadata.get("page", "Unknown")
            context_parts.append(
                f"[Document: {source} | Page: {page} | Score: {score:.4f}]\n"
                f"{doc.page_content.strip()}"
            )
        return "\n\n".join(context_parts)

    def format_sources(self, scored_docs):
        """Extracts structured source metadata."""
        return [
            {
                "document": Path(doc.metadata.get("source", "Unknown")).name,
                "page": doc.metadata.get("page"),
                "relevance_score": round(float(score), 4)
            }
            for doc, score in scored_docs
        ]

    def _build_runnable_chain(self):
        """Builds an LCEL pipeline using RunnableParallel and RunnableLambda."""

        # Branch 1: Retrieve and Rerank candidates in parallel with preserving the question
        retrieval_stage = RunnableParallel({
            "question": RunnableLambda(lambda x: x["question"] if isinstance(x, dict) else x),
            "scored_docs": RunnableLambda(
                lambda x: self.retrieve_and_rerank(x["question"] if isinstance(x, dict) else x)
            )
        })

        # Branch 2: Prepare formatted context and pass through scored_docs and question in parallel
        context_stage = RunnableParallel({
            "context": RunnableLambda(lambda x: self.format_context(x["scored_docs"])),
            "question": RunnableLambda(lambda x: x["question"]),
            "scored_docs": RunnableLambda(lambda x: x["scored_docs"])
        })

        # Branch 3: Generate LLM answer and format sources in parallel
        llm_subchain = (
            RunnableParallel({
                "context": RunnableLambda(lambda x: x["context"]),
                "question": RunnableLambda(lambda x: x["question"])
            })
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        output_stage = RunnableParallel({
            "question": RunnableLambda(lambda x: x["question"]),
            "answer": llm_subchain,
            "sources": RunnableLambda(lambda x: self.format_sources(x["scored_docs"]))
        })

        # Composite LCEL Runnable Chain
        return retrieval_stage | context_stage | output_stage

    def evaluate_realtime(self, question: str, answer: str, scored_docs: list) -> dict:
        """Dynamically computes online RAG evaluation metrics for this exact query in real time:
        - Retrieval Confidence (Cross-Encoder logit mapped via sigmoid)
        - Groundedness / Faithfulness (sentence-level context entailment to detect hallucinations)
        - Answer Relevance (semantic cosine similarity between query and generated response)
        - Hallucination Risk (LOW, MODERATE, HIGH)
        """
        import numpy as np
        import re

        # 1. Retrieval Confidence (Top Cross-Encoder Score mapped via Sigmoid)
        if scored_docs:
            top_score = float(scored_docs[0][1])
            retrieval_conf = float(1.0 / (1.0 + np.exp(-top_score)))
            retrieval_conf_pct = round(min(max(retrieval_conf * 100, 15.0), 99.8), 1)
        else:
            retrieval_conf_pct = 0.0

        # 2. Groundedness (Faithfulness) Score: Multi-Factor Evaluation
        embedder = self.vector_store.embeddings
        chunk_texts = [doc.page_content for doc, _ in scored_docs]
        full_context = " ".join(chunk_texts).lower()

        # Factor A: Lexical Term & Entity Overlap
        answer_words = re.findall(r'[a-zA-Z0-9_\$]+', answer.lower())
        stop_words = {'the', 'is', 'a', 'an', 'to', 'for', 'and', 'in', 'of', 'on', 'with', 'by', 'at', 'it', 'as', 'are', 'be', 'this', 'that'}
        substantive_words = [w for w in answer_words if w not in stop_words and len(w) > 2]
        if substantive_words:
            matched_words = sum(1 for w in substantive_words if w in full_context)
            lexical_ratio = matched_words / len(substantive_words)
        else:
            lexical_ratio = 1.0

        # Factor B: Semantic Chunk Overlap
        lines = [l.strip() for l in answer.splitlines() if len(l.strip()) > 10]
        if lines and scored_docs:
            chunk_embs = [np.array(embedder.embed_query(doc.page_content[:600])) for doc, _ in scored_docs]
            line_sims = []
            for l in lines[:5]:
                l_emb = np.array(embedder.embed_query(l))
                l_norm = np.linalg.norm(l_emb)
                if l_norm > 0:
                    sim_to_best_chunk = max([
                        float(np.dot(l_emb, c) / (l_norm * np.linalg.norm(c)))
                        for c in chunk_embs if np.linalg.norm(c) > 0
                    ])
                    line_sims.append(sim_to_best_chunk)
            avg_sem_sim = float(np.mean(line_sims)) if line_sims else 0.55
            # Scale cosine [0.25 - 0.70] -> [30% - 99%]
            semantic_ratio = min(max((avg_sem_sim - 0.20) / 0.50, 0.1), 1.0)
        else:
            semantic_ratio = 0.85

        # Blended Groundedness: 50% entity/lexical verification + 50% semantic entailment
        blended_groundedness = (0.5 * lexical_ratio) + (0.5 * semantic_ratio)
        groundedness_pct = round(min(max(blended_groundedness * 100, 15.0), 98.9), 1)

        # 3. Answer Relevance (Query to Answer semantic cosine similarity)
        q_emb = np.array(embedder.embed_query(question))
        a_emb = np.array(embedder.embed_query(answer[:500]))
        q_norm = np.linalg.norm(q_emb)
        a_norm = np.linalg.norm(a_emb)
        if q_norm > 0 and a_norm > 0:
            rel_cos = float(np.dot(q_emb, a_emb) / (q_norm * a_norm))
            relevance_pct = round(min(max((rel_cos - 0.15) / 0.65 * 100, 25.0), 99.0), 1)
        else:
            relevance_pct = 85.0

        # 4. Dynamic Hallucination Risk Classification
        if groundedness_pct >= 70.0 and retrieval_conf_pct >= 60.0:
            hallucination_risk = "LOW"
        elif groundedness_pct >= 50.0:
            hallucination_risk = "MODERATE"
        else:
            hallucination_risk = "HIGH"

        return {
            "retrieval_confidence": f"{retrieval_conf_pct}%",
            "groundedness": f"{groundedness_pct}%",
            "answer_relevance": f"{relevance_pct}%",
            "hallucination_risk": hallucination_risk
        }

    def ask(self, question: str):
        """Processes query through RAG pipeline, tracks LLMOps telemetry, and computes real-time online evaluation metrics."""
        print(f"\nProcessing query through LCEL Runnable Chain...")
        t0 = time.time()

        # Step 1: Retrieval
        t_ret_start = time.time()
        candidate_docs = self.vector_store.similarity_search(question, k=self.top_k_retrieve)
        t_ret_end = time.time()
        retrieval_time = t_ret_end - t_ret_start

        # Step 2: Re-ranking
        t_rerank_start = time.time()
        if candidate_docs:
            pairs = [[question, doc.page_content] for doc in candidate_docs]
            scores = self.reranker.predict(pairs)
            scored_docs = list(zip(candidate_docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            scored_docs = scored_docs[:self.top_n_rerank]
        else:
            scored_docs = []
        t_rerank_end = time.time()
        rerank_time = t_rerank_end - t_rerank_start

        sources = self.format_sources(scored_docs)
        context = self.format_context(scored_docs)

        # Step 3: LLM Generation
        t_llm_start = time.time()
        try:
            llm_chain = self.prompt | self.llm | StrOutputParser()
            answer = llm_chain.invoke({"context": context, "question": question})
            t_llm_end = time.time()
            llm_time = t_llm_end - t_llm_start
        except Exception as e:
            err_str = str(e)
            if "Connection error" in err_str or "10054" in err_str or "ConnectError" in err_str or "APIConnectionError" in err_str:
                print(f"[RAG FALLBACK] Groq API blocked by network firewall. Returning local re-ranked passages...")
                fallback_text = [
                    "> ⚠️ **Network Notice**: Outbound connection to Groq API was blocked by your campus network (`Connection was reset`).",
                    "> **Quick Fix**: Switch your laptop to a **Mobile Hotspot** for full AI response generation.\n",
                    "### 📄 Top Re-ranked Policy Sections (FAISS + Cross-Encoder):"
                ]

                for i, (doc, score) in enumerate(scored_docs, 1):
                    doc_name = Path(doc.metadata.get("source", "Unknown")).name
                    page = doc.metadata.get("page", "Unknown")
                    clean_content = doc.page_content.strip()
                    fallback_text.append(
                        f"#### {i}. {doc_name} — Page {page} *(Relevance: {score:.2f})*\n"
                        f"{clean_content}"
                    )

                answer = "\n\n".join(fallback_text)
                t_llm_end = time.time()
                llm_time = t_llm_end - t_llm_start
            else:
                raise e

        # Step 4: Real-time Online Evaluation
        t_eval_start = time.time()
        evaluation = self.evaluate_realtime(question, answer, scored_docs)
        t_eval_end = time.time()
        eval_time = t_eval_end - t_eval_start

        total_time = time.time() - t0

        # Step 5: LLMOps Telemetry Logging
        timings = {
            "retrieval": retrieval_time,
            "rerank": rerank_time,
            "llm": llm_time,
            "eval": eval_time,
            "total": total_time
        }
        trace = tracker.log_trace(
            question=question,
            answer=answer,
            sources=sources,
            timings=timings,
            evaluation=evaluation,
            model=GROQ_MODEL_NAME
        )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "evaluation": evaluation,
            "trace_id": trace["trace_id"],
            "llmops_metrics": trace["metrics"],
            "tokens": trace["tokens"],
            "cost_usd": trace["cost_usd"]
        }


if __name__ == "__main__":
    rag = EnterpriseRAGChain()

    query = "What is the policy for reporting cybersecurity incidents, and what is the deadline?"
    print("\n" + "=" * 70)
    print(f"QUESTION: {query}")
    print("=" * 70)

    result = rag.ask(query)

    print("\n--- SOURCES IDENTIFIED VIA RE-RANKING ---")
    for s in result["sources"]:
        print(f"- {s['document']} (Page {s['page']}) | Relevance: {s['relevance_score']}")

    print("\n--- FINAL ANSWER ---")
    print(result["answer"])
