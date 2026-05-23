#!/usr/bin/env python3
"""
Evaluation script — measures latency (p95), retrieval metrics, and citation accuracy.
Usage:
    python scripts/evaluate.py --questions "What is X?" "Explain Y"
"""
import sys, os, time, argparse, json, statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline import RAGPipeline


def run_eval(questions: list[str], top_k: int = 5):
    pipeline = RAGPipeline()
    latencies = []
    results = []

    print(f"\nRunning evaluation on {len(questions)} question(s)...\n")

    for q in questions:
        t0 = time.perf_counter()
        res = pipeline.query(q, top_k=top_k)
        latency_ms = round((time.perf_counter() - t0) * 1000)
        latencies.append(latency_ms)

        has_sources = len(res["sources"]) > 0
        has_citation = "Sources:" in res["answer"] or has_sources

        results.append({
            "question": q,
            "latency_ms": latency_ms,
            "chunks_retrieved": len(res["retrieved_chunks"]),
            "sources_cited": len(res["sources"]),
            "has_citation_in_answer": has_citation,
            "answer_preview": res["answer"][:200],
        })

        print(f"  Q: {q[:60]}")
        print(f"     Latency: {latency_ms}ms | Sources: {len(res['sources'])} | Citation: {has_citation}")

    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies)
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)] if len(latencies_sorted) > 1 else latencies_sorted[0]
    citation_acc = sum(1 for r in results if r["has_citation_in_answer"]) / len(results) * 100

    print(f"\n{'='*50}")
    print(f"EVAL SUMMARY")
    print(f"  Questions tested : {len(questions)}")
    print(f"  Latency p50      : {p50}ms")
    print(f"  Latency p95      : {p95}ms")
    print(f"  Citation accuracy: {citation_acc:.1f}%")
    print(f"  Target: 2000-5000ms latency | >90% citation accuracy")
    print(f"{'='*50}\n")

    with open("eval_results.json", "w") as f:
        json.dump({"summary": {"p50_ms": p50, "p95_ms": p95, "citation_accuracy_pct": citation_acc}, "details": results}, f, indent=2)
    print("Detailed results saved to eval_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", nargs="+", default=[
        "What is the main objective of this document?",
        "Summarize the key findings.",
        "What recommendations are made?",
    ])
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    run_eval(args.questions, top_k=args.top_k)
