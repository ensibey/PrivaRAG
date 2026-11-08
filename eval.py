"""CLI evaluation runner for measuring faithfulness, answer relevance, and latency."""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, List

from langchain_ollama import ChatOllama

from config import LLM_MODEL, OLLAMA_BASE_URL
from rag_engine import query as rag_query

DEFAULT_BENCHMARK: List[Dict[str, str]] = [
    {
        "question": "What is the main topic of the uploaded documents?",
        "ground_truth": "General overview of the indexed corpus.",
    },
    {
        "question": "Summarize the key financial figures or metrics mentioned.",
        "ground_truth": "Specific numbers, dates, and financial metrics from the files.",
    },
    {
        "question": "What are the main conclusions or recommendations?",
        "ground_truth": "Action items or conclusions listed in the document.",
    },
]

FAITHFULNESS_PROMPT = """Evaluate whether the ANSWER is strictly grounded in the CONTEXT.
Score from 0.0 (hallucinated) to 1.0 (fully supported by context).
Return ONLY a single floating-point number between 0.0 and 1.0.

CONTEXT:
{context}

ANSWER:
{answer}

Score:"""

RELEVANCE_PROMPT = """Evaluate how directly the ANSWER addresses the QUESTION.
Score from 0.0 (irrelevant) to 1.0 (complete and direct).
Return ONLY a single floating-point number between 0.0 and 1.0.

QUESTION:
{question}

ANSWER:
{answer}

Score:"""


def _parse_score(raw: str) -> float:
    try:
        token = raw.strip().split()[0].replace(",", ".")
        val = float(token)
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5


def evaluate_sample(llm: ChatOllama, question: str) -> Dict[str, object]:
    """Run a single query and compute faithfulness and relevance scores."""
    t0 = time.perf_counter()
    res = rag_query(question, chat_history=[], use_cache=False)
    latency_s = time.perf_counter() - t0

    answer = res.get("answer", "")
    sources = res.get("sources", [])
    context_str = "\n---\n".join(s.get("text", "") for s in sources)

    if not context_str.strip():
        return {
            "question": question,
            "answer": answer,
            "faithfulness": 0.0,
            "relevance": 0.0,
            "chunks_retrieved": 0,
            "latency_s": round(latency_s, 2),
        }

    f_raw = llm.invoke(
        FAITHFULNESS_PROMPT.format(context=context_str, answer=answer)
    ).content
    r_raw = llm.invoke(
        RELEVANCE_PROMPT.format(question=question, answer=answer)
    ).content

    return {
        "question": question,
        "answer": answer,
        "faithfulness": _parse_score(f_raw),
        "relevance": _parse_score(r_raw),
        "chunks_retrieved": len(sources),
        "latency_s": round(latency_s, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PrivaRAG retrieval and answer quality.")
    parser.add_argument("--dataset", type=str, help="Path to JSON benchmark file")
    parser.add_argument("--output", type=str, help="Optional JSON output path")
    args = parser.parse_args()

    samples = DEFAULT_BENCHMARK
    if args.dataset:
        with open(args.dataset, "r", encoding="utf-8") as f:
            samples = json.load(f)

    judge = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)
    results = []

    print(f"Running evaluation on {len(samples)} queries (judge={LLM_MODEL})...")
    print("-" * 72)

    for idx, item in enumerate(samples, 1):
        q = item["question"]
        metrics = evaluate_sample(judge, q)
        results.append(metrics)
        print(
            f"[{idx}/{len(samples)}] faithfulness={metrics['faithfulness']:.2f} "
            f"relevance={metrics['relevance']:.2f} "
            f"chunks={metrics['chunks_retrieved']} "
            f"latency={metrics['latency_s']:.2f}s | {q[:45]}"
        )

    n = max(len(results), 1)
    avg_f = sum(float(r["faithfulness"]) for r in results) / n
    avg_r = sum(float(r["relevance"]) for r in results) / n
    avg_l = sum(float(r["latency_s"]) for r in results) / n

    print("-" * 72)
    print(
        f"Summary: mean_faithfulness={avg_f:.2f} | mean_relevance={avg_r:.2f} | mean_latency={avg_l:.2f}s"
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
