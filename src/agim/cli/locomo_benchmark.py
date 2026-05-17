"""LoCoMo retrieval benchmark for AGIM Path A.

Indexes dialog turns into FAISS+BM25, then retrieves relevant context
for each QA question. Measures hit rate: does retrieved context contain answer?
"""
import json, time
from collections import defaultdict
from pathlib import Path

from agim.memory.faiss_retrieval import FAISSRetrieval

LOCOMO = "/mnt/hf_model_weights/arman/3bit/sites/locomo/data/locomo10.json"


def load_locomo(path: str = LOCOMO) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def index_conversation(conv: dict, faiss: FAISSRetrieval) -> int:
    """Index turns + observations + summaries."""
    count = 0
    dialogue = conv["conversation"]
    # Raw turns
    for key in sorted(dialogue.keys()):
        if key.startswith("session_") and isinstance(dialogue[key], list):
            for turn in dialogue[key]:
                if isinstance(turn, dict) and "text" in turn:
                    s, d = turn.get("speaker", "?"), turn.get("dia_id", "?")
                    t = turn["text"]
                    faiss.add(f"[{d}] {s}: {t[:150]}", f"[{d}] {s}: {t}")
                    count += 1
    # Observations
    for key, val in conv.get("observation", {}).items():
        if isinstance(val, str):
            faiss.add(f"{key}: {val[:150]}", val); count += 1
    # Summaries
    for key, val in conv.get("session_summary", {}).items():
        if isinstance(val, str):
            faiss.add(f"{key}: {val[:150]}", val); count += 1
    return count


def run_locomo_benchmark(data: list[dict], n_conv: int = 10) -> dict:
    faiss = FAISSRetrieval(dim=768)
    results = {"total_qa": 0, "retrieved": 0, "correct": 0,
               "by_category": defaultdict(lambda: {"correct": 0, "total": 0, "retrieved": 0}),
               "by_sample": {}, "index_time_s": 0, "eval_time_s": 0}

    # Phase 1: Index all conversations
    print("Indexing conversations...")
    t0 = time.time()
    total_turns = 0
    for sample in data[:n_conv]:
        total_turns += index_conversation(sample, faiss)
    faiss.build_index()
    results["index_time_s"] = time.time() - t0
    print(f"  {total_turns} turns indexed in {results['index_time_s']:.1f}s")

    # Phase 2: Answer questions via retrieval
    print(f"\nEvaluating on {n_conv} conversations...\n")
    t1 = time.time()
    for sample in data[:n_conv]:
        sid = sample.get("sample_id", "?")
        qa_pairs = sample.get("qa", [])
        s_correct = s_retrieved = 0

        for qa in qa_pairs:
            question = qa["question"]
            expected = str(qa.get("answer") or qa.get("adversarial_answer", "")).lower()
            cat = qa.get("category", 0)

            # Search FAISS for relevant turns
            hits = faiss.search(question, top_k=5)
            found = False
            for hit in hits:
                if expected in hit["value"].lower():
                    found = True
                    break

            results["total_qa"] += 1
            results["by_category"][cat]["total"] += 1
            if found:
                results["retrieved"] += 1
                results["correct"] += 1
                s_correct += 1
                results["by_category"][cat]["correct"] += 1
                results["by_category"][cat]["retrieved"] += 1

        total = len(qa_pairs)
        results["by_sample"][sid] = {
            "qa_total": total, "retrieved": s_correct,
            "retrieval_rate": round(s_correct / max(total, 1), 4),
        }
        print(f"  {sid}: {s_correct}/{total} ({s_correct/max(total,1):.1%})")

    results["eval_time_s"] = time.time() - t1
    results["retrieval_rate"] = round(results["retrieved"] / max(results["total_qa"], 1), 4)
    results["by_category"] = {str(k): {"rate": round(v["correct"] / max(v["total"], 1), 4),
                                        "retrieved": v["correct"], "total": v["total"]}
                              for k, v in results["by_category"].items()}
    return results


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--output", default="locomo_results.json")
    args = p.parse_args()

    print(f"Loading LoCoMo...")
    data = load_locomo()
    print(f"  {len(data)} conversations, {sum(len(s['qa']) for s in data)} QA pairs\n")

    results = run_locomo_benchmark(data, n_conv=args.n_samples)

    print(f"\n{'='*60}")
    print(f"LOCOMO RETRIEVAL BENCHMARK — AGIM Path A (FAISS+BM25)")
    print(f"{'='*60}")
    print(f"  Total QA:        {results['total_qa']}")
    print(f"  Retrieved answer: {results['retrieved']}")
    print(f"  Retrieval rate:   {results['retrieval_rate']:.1%}")
    print(f"  Index time:      {results['index_time_s']:.1f}s")
    print(f"  Eval time:       {results['eval_time_s']:.1f}s")
    print(f"\n  By category:")
    for cat, s in sorted(results["by_category"].items()):
        print(f"    Cat {cat}: {s['rate']:.1%} ({s['retrieved']}/{s['total']})")
    print(f"\n  By sample:")
    for sid, s in results["by_sample"].items():
        print(f"    {sid}: {s['retrieval_rate']:.1%} ({s['retrieved']}/{s['qa_total']})")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
