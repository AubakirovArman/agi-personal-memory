#!/usr/bin/env python3
"""LoCoMo 50-step plan — all 5 phases in one benchmark."""
import json, time, re, math, sys, os
import numpy as np
from collections import defaultdict, Counter
from openai import OpenAI
from sentence_transformers import SentenceTransformer

LOC = "/mnt/hf_model_weights/arman/3bit/sites/locomo/data/locomo10.json"
FACTS_CACHE = "locomo_extracted_facts.json"

def main():
    with open(LOC) as f: data = json.load(f)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY to run the LoCoMo full benchmark.")
    client = OpenAI()
    emb_model = SentenceTransformer('all-mpnet-base-v2')

    # ───── PHASE 1: LLM Atomic Fact Extraction ─────
    print("=" * 60)
    print("PHASE 1: LLM-based atomic fact extraction")
    print("=" * 60)

    if os.path.exists(FACTS_CACHE):
        with open(FACTS_CACHE) as f: cache = json.load(f)
        all_fact_texts = cache["texts"]
        all_fact_contexts = cache["contexts"]
        print(f"  Loaded {len(all_fact_texts)} cached facts")
    else:
        all_fact_texts, all_fact_contexts = [], []
        total_turns = 0

        for si, sample in enumerate(data):
            dialogue = sample["conversation"]
            sample_facts = 0
            for key in sorted(dialogue.keys()):
                if not key.startswith("session_") or not isinstance(dialogue[key], list):
                    continue
                for turn in dialogue[key]:
                    if not isinstance(turn, dict) or "text" not in turn: continue
                    text, speaker = turn["text"], turn.get("speaker", "?")
                    turn_id = turn.get("dia_id", "?")
                    total_turns += 1
                    if len(text.strip()) < 30: continue

                    prompt = f'Extract ALL atomic facts from this dialog turn. Return JSON array of strings. Each string is one atomic fact (concise statement).\\nTurn: {text[:400]}\\nFacts (JSON array):'
                    try:
                        r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=200, temperature=0.0)
                        resp = r.choices[0].message.content.strip()
                        match = re.search(r'\[.*\]', resp, re.DOTALL)
                        facts = json.loads(match.group()) if match else []
                    except:
                        facts = [text[:150]]

                    for ft in (facts if isinstance(facts, list) else [str(facts)]):
                        ft = str(ft).strip()
                        if len(ft) < 5: continue
                        all_fact_texts.append(ft)
                        all_fact_contexts.append(f"{ft} ({speaker}, {turn_id})")
                        sample_facts += 1

            print(f"  {sample['sample_id']}: {sample_facts} facts", flush=True)

        with open(FACTS_CACHE, "w") as f:
            json.dump({"texts": all_fact_texts, "contexts": all_fact_contexts}, f)
        print(f"  TOTAL: {len(all_fact_texts)} atomic facts from {total_turns} turns\n")

    # ───── PHASE 2: Multi-Signal Retrieval ─────
    print("=" * 60)
    print("PHASE 2: Multi-signal retrieval index")
    print("=" * 60)

    class BM25Idx:
        def __init__(self, docs):
            self.docs, self.N = docs, len(docs)
            self.lens = [len(d.split()) for d in docs]
            self.avgdl = sum(self.lens)/max(self.N,1)
            self.df = Counter()
            for d in docs:
                for w in set(d.lower().split()): self.df[w] += 1
        def scores(self, q):
            s = np.zeros(self.N)
            for w in q.lower().split():
                if w not in self.df: continue
                idf = math.log((self.N-self.df[w]+0.5)/(self.df[w]+0.5)+1)
                for i, d in enumerate(self.docs):
                    tf = d.lower().split().count(w)
                    if tf: s[i] += idf*(tf*2.5)/(tf+1.5*(1-0.25+0.25*self.lens[i]/self.avgdl))
            return s

    bm25 = BM25Idx(all_fact_texts)
    doc_embs = emb_model.encode(all_fact_texts, show_progress_bar=True, convert_to_numpy=True)
    print(f"  {len(doc_embs)} vectors x {doc_embs.shape[1]}d\n")

    def retrieve(query, top_k=8, w_dense=0.8):
        q_emb = emb_model.encode([query], convert_to_numpy=True)[0]
        dense = np.dot(doc_embs, q_emb)
        dense_n = dense / (np.max(np.abs(dense)) + 1e-8)
        bm25_s = bm25.scores(query)
        bm25_n = bm25_s / (np.max(bm25_s) + 1e-8)
        fused = w_dense * dense_n + (1-w_dense) * bm25_n
        top = np.argsort(fused)[-top_k:][::-1]
        return [all_fact_contexts[i] for i in top], [all_fact_texts[i] for i in top]

    # ───── PHASE 3: Reasoning ─────
    print("=" * 60)
    print("PHASE 3-4-5: Enhanced generation + LLM-as-judge")
    print("=" * 60)

    FEWSHOT = """Example 1:
Facts: - John graduated from Stanford in 2019\n- John works at Google since 2020
Question: Where did John graduate from?
Answer: Stanford.

Example 2:
Facts: - Mary visited Paris in June 2022\n- After Paris, Mary went to London\n- Mary returned home in August
Question: What did Mary do after visiting Paris?
Answer: She went to London.

Example 3:
Facts: - The charity event was on May 8th 2023\n- 200 people attended\n- They raised $5000 for cancer research
Question: When was the charity event?
Answer: May 8th 2023."""

    GEN_PROMPT = FEWSHOT + """\n\nNow answer based on these facts. If the answer is not in the facts, say EXACTLY "Not mentioned".

FACTS:
{context}

Question: {question}
Answer (concise):"""

    def norm(s): return ' '.join(re.sub(r'[,.!?;:()\[\]{}"\']', ' ', s.lower().strip()).split())
    def f1(p, g):
        pt, gt = norm(p).split(), norm(g).split()
        if not pt or not gt: return 0.0
        c = set(pt) & set(gt)
        if not c: return 0.0
        pr, rc = len(c)/len(pt), len(c)/len(gt)
        return 2*pr*rc/(pr+rc)

    def llm_judge(question, generated, gold):
        prompt = f'Evaluate if the generated answer correctly answers the question. Consider semantic equivalence.\nQuestion: {question}\nGold Answer: {gold}\nGenerated Answer: {generated}\nIs the generated answer correct? Reply ONLY YES or NO.'
        try:
            r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=5, temperature=0.0)
            return 1.0 if r.choices[0].message.content.strip().upper() == "YES" else 0.0
        except:
            return f1(generated, gold)

    total_qa = 0
    total_f1 = 0.0
    total_judge = 0.0
    by_cat = defaultdict(lambda: {"f1": 0.0, "judge": 0.0, "total": 0})
    by_sample = {}
    t0 = time.time()

    for si, sample in enumerate(data):
        sid = sample["sample_id"]
        s_f1, s_judge = 0.0, 0.0
        for qa in sample["qa"]:
            q = qa["question"]
            gold = str(qa.get("answer") or qa.get("adversarial_answer", ""))
            cat = qa.get("category", 0)

            _ctx_texts, _ = retrieve(q, top_k=8)
            context_str = "\n".join(_ctx_texts)[:4000]
            prompt = GEN_PROMPT.replace('{context}', context_str).replace('{question}', q)

            try:
                r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=80, temperature=0.0)
                pred = r.choices[0].message.content.strip()
            except:
                pred = "Not mentioned"

            # F1 score
            if cat in (2,3,4): sc_f1 = f1(pred, gold)
            elif cat == 1: sc_f1 = f1(pred, gold.split(";")[0].strip())
            elif cat == 5: sc_f1 = 1.0 if "no information available" in pred.lower() or "not mentioned" in pred.lower() else 0.0
            else: sc_f1 = f1(pred, gold)

            # LLM-as-judge
            sc_judge = llm_judge(q, pred, gold)

            total_qa += 1; total_f1 += sc_f1; total_judge += sc_judge
            s_f1 += sc_f1; s_judge += sc_judge
            by_cat[cat]["total"] += 1; by_cat[cat]["f1"] += sc_f1; by_cat[cat]["judge"] += sc_judge

            if total_qa <= 5 or total_qa % 500 == 0:
                e = time.time()-t0
                eta = e/total_qa*(1986-total_qa) if total_qa else 0
                print(f"  [{total_qa}/1986 {e:.0f}s ETA{eta:.0f}s] cat={cat} F1={sc_f1:.2f} Judge={sc_judge:.0f} | {pred[:50]}...", flush=True)

        by_sample[sid] = {"f1": round(s_f1/max(len(sample["qa"]),1),4),
                          "judge": round(s_judge/max(len(sample["qa"]),1),4)}
        print(f"  {sid}: F1={s_f1/max(len(sample['qa']),1):.1%} Judge={s_judge/max(len(sample['qa']),1):.1%}", flush=True)

    e = time.time()-t0
    acc_f1 = total_f1/total_qa
    acc_judge = total_judge/total_qa

    print(f"\n{'='*60}")
    print(f"LOCOMO FINAL — All 5 Phases")
    print(f"{'='*60}")
    print(f"  Facts: {len(all_fact_texts)}  QA: {total_qa}  Time: {e:.0f}s ({e/60:.1f}min)")
    print(f"  F1 Score:     {acc_f1:.1%}")
    print(f"  LLM Judge:    {acc_judge:.1%}")
    print(f"\n  By Category (F1 / Judge):")
    for cat, s in sorted(by_cat.items()):
        print(f"    Cat {cat}: F1={s['f1']/max(s['total'],1):.1%}  Judge={s['judge']/max(s['total'],1):.1%}")
    print(f"\n  By Sample:")
    for sid, s in sorted(by_sample.items()):
        print(f"    {sid}: F1={s['f1']:.1%}  Judge={s['judge']:.1%}")

    with open("locomo_50steps_results.json", "w") as f:
        json.dump({"accuracy_f1": round(acc_f1,4), "accuracy_judge": round(acc_judge,4),
                   "facts": len(all_fact_texts), "total_qa": total_qa, "time_s": round(e,1),
                   "by_category": {str(k): {"f1": round(v["f1"]/max(v["total"],1),4),
                                            "judge": round(v["judge"]/max(v["total"],1),4)}
                                  for k,v in by_cat.items()},
                   "by_sample": by_sample}, f, indent=2)
    print(f"\nSaved to locomo_50steps_results.json")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
