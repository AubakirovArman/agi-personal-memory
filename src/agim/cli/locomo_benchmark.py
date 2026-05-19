"""LoCoMo benchmark — AGIM Path A with dense embeddings + FAISS."""
import json, time, os
import numpy as np
from collections import defaultdict

from sentence_transformers import SentenceTransformer

LOCOMO = os.environ.get("AGIM_LOCOMO_PATH", "sites/locomo/data/locomo10.json")


def load_locomo(path: str = LOCOMO) -> list[dict]:
    with open(path) as f: return json.load(f)


class DenseRetriever:
    """Dense retrieval with sentence-transformer embeddings."""
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension() if hasattr(self.model, 'get_sentence_embedding_dimension') else 768
        self.embeddings: list[np.ndarray] = []
        self.texts: list[str] = []
        self.full_texts: list[str] = []

    def index_conversation(self, conv: dict) -> int:
        count = 0
        dialogue = conv["conversation"]
        for key in sorted(dialogue.keys()):
            if key.startswith("session_") and isinstance(dialogue[key], list):
                for turn in dialogue[key]:
                    if isinstance(turn, dict) and "text" in turn:
                        s = turn.get("speaker", "?")
                        d = turn.get("dia_id", "?")
                        t = turn["text"]
                        self.texts.append(f"[{d}] {s}: {t[:300]}")
                        self.full_texts.append(t)
                        count += 1
        for key, val in conv.get("observation", {}).items():
            if isinstance(val, str):
                self.texts.append(val[:300])
                self.full_texts.append(val); count += 1
        for key, val in conv.get("session_summary", {}).items():
            if isinstance(val, str):
                self.texts.append(val[:300])
                self.full_texts.append(val); count += 1
        return count

    def build_index(self):
        print(f"  Encoding {len(self.texts)} documents...")
        self.embeddings = self.model.encode(self.texts, show_progress_bar=True,
                                             convert_to_numpy=True)
        print(f"  Index built: {len(self.embeddings)} vectors x {self.dim}d")

    def search(self, query: str, top_k: int = 5) -> list[str]:
        q_emb = self.model.encode([query], convert_to_numpy=True)[0]
        scores = np.dot(self.embeddings, q_emb)
        top_idx = np.argsort(scores)[-top_k:][::-1]
        return [self.full_texts[i] for i in top_idx]


def run_locomo_benchmark(data: list[dict], n_conv: int = 10,
                         model_name: str = "all-mpnet-base-v2") -> dict:
    retriever = DenseRetriever(model_name=model_name)
    results = {"total_qa": 0, "retrieved": 0,
               "by_category": defaultdict(lambda: {"retrieved": 0, "total": 0}),
               "by_sample": {}, "index_time_s": 0, "eval_time_s": 0}

    print("Indexing conversations...")
    t0 = time.time()
    total = 0
    for sample in data[:n_conv]:
        total += retriever.index_conversation(sample)
    retriever.build_index()
    results["index_time_s"] = time.time() - t0
    print(f"  {total} docs indexed in {results['index_time_s']:.1f}s\n")

    print(f"Evaluating on {n_conv} conversations...\n")
    t1 = time.time()
    for sample in data[:n_conv]:
        sid = sample.get("sample_id", "?")
        qa_pairs = sample.get("qa", [])
        s_correct = 0
        for qa in qa_pairs:
            question = qa["question"]
            expected = str(qa.get("answer") or qa.get("adversarial_answer", "")).lower()
            cat = qa.get("category", 0)
            hits = retriever.search(question, top_k=5)
            found = any(expected in h.lower() for h in hits)
            results["total_qa"] += 1
            results["by_category"][cat]["total"] += 1
            if found:
                results["retrieved"] += 1
                results["by_category"][cat]["retrieved"] += 1
                s_correct += 1
        total = len(qa_pairs)
        results["by_sample"][sid] = {
            "qa_total": total, "retrieved": s_correct,
            "rate": round(s_correct / max(total, 1), 4),
        }
        print(f"  {sid}: {s_correct}/{total} ({s_correct/max(total,1):.1%})")

    results["eval_time_s"] = time.time() - t1
    results["rate"] = round(results["retrieved"] / max(results["total_qa"], 1), 4)
    results["by_category"] = {str(k): {"rate": round(v["retrieved"] / max(v["total"], 1), 4),
                                        "retrieved": v["retrieved"], "total": v["total"]}
                              for k, v in results["by_category"].items()}
    return results


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--model", default="all-mpnet-base-v2")
    p.add_argument(
        "--output",
        default="results/memory_retrieval/locomo_dense_results.json",
    )
    args = p.parse_args()
    print(f"Loading LoCoMo...")
    data = load_locomo()
    qa_total = sum(len(s['qa']) for s in data)
    print(f"  {len(data)} conversations, {qa_total} QA pairs\n")

    results = run_locomo_benchmark(data, n_conv=args.n_samples, model_name=args.model)

    print(f"\n{'='*60}")
    print(f"LOCOMO — AGIM Dense Retrieval (all-MiniLM-L6-v2)")
    print(f"{'='*60}")
    print(f"  QA pairs:        {results['total_qa']}")
    print(f"  Retrieved:       {results['retrieved']}")
    print(f"  Retrieval rate:  {results['rate']:.1%}")
    print(f"  Index: {results['index_time_s']:.1f}s  Eval: {results['eval_time_s']:.1f}s")
    print(f"\n  By category:")
    for cat, s in sorted(results["by_category"].items()):
        print(f"    Cat {cat}: {s['rate']:.1%} ({s['retrieved']}/{s['total']})")
    print(f"\n  By sample:")
    for sid, s in sorted(results["by_sample"].items()):
        print(f"    {sid}: {s['rate']:.1%} ({s['retrieved']}/{s['qa_total']})")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
