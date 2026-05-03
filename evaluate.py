import json
import math
import os
import pandas as pd

# ----------------------------
# Load qrels
# ----------------------------
qrels = {}
with open("dataset/qrels/test.tsv", "r") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 3: continue
        qid, docid, rel = parts[0], parts[1], parts[2]
        qrels.setdefault(qid, {})[docid] = int(rel)

print(f"Qrels: {len(qrels)} queries\n")

# ----------------------------
# Metric helpers
# ----------------------------
def dcg(rels):
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))

def ndcg(pred, q, k):
    gains = [q.get(d, 0) for d, _ in pred[:k]]
    idcg  = dcg(sorted(q.values(), reverse=True)[:k])
    return dcg(gains) / idcg if idcg > 0 else 0

def precision(pred, q, k):
    return sum(1 for d, _ in pred[:k] if q.get(d, 0) > 0) / k if k else 0

def avg_precision(pred, q):
    hits = score = 0
    for i, (d, _) in enumerate(pred):
        if q.get(d, 0) > 0:
            hits += 1; score += hits / (i + 1)
    return score / max(len(q), 1)

# ----------------------------
# Load and evaluate all runs
# ----------------------------
def load_run(path):
    run = {}
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 6: continue
            run.setdefault(p[0], []).append((p[2], float(p[4])))
    return {qid: sorted(docs, key=lambda x: x[1], reverse=True)
            for qid, docs in run.items()}

# Auto-discover all .trec files in output/
trec_files = sorted(f for f in os.listdir("output") if f.endswith(".trec"))

# Fixed display order
ORDER = ["run_bm25.trec", "run_lm.trec", "run_pagerank.trec",
         "run_fused_bm25_dom.trec", "run_fused_no_pr.trec",
         "run_fused_rrf_60.trec", "run_fused_rrf_10.trec", "run_fused.trec"]
trec_files = [f for f in ORDER if f in trec_files] + \
             [f for f in trec_files if f not in ORDER]

metrics = {}
for fname in trec_files:
    path = os.path.join("output", fname)
    run  = load_run(path)
    tag  = fname.replace("run_", "").replace(".trec", "").upper()

    map_s, n10, n20, p10, p20 = [], [], [], [], []
    for qid, pred in run.items():
        q = qrels.get(qid, {})
        map_s.append(avg_precision(pred, q))
        n10.append(ndcg(pred, q, 10))
        n20.append(ndcg(pred, q, 20))
        p10.append(precision(pred, q, 10))
        p20.append(precision(pred, q, 20))

    n = len(map_s)
    metrics[tag] = {
        "MAP":     sum(map_s) / n,
        "nDCG@10": sum(n10)   / n,
        "nDCG@20": sum(n20)   / n,
        "P@10":    sum(p10)   / n,
        "P@20":    sum(p20)   / n,
    }

df = pd.DataFrame(metrics).T[["MAP", "nDCG@10", "nDCG@20", "P@10", "P@20"]]

print("=== FINAL RESULTS ===\n")
print(df.to_string(float_format=lambda x: f"{x:.6f}"))

#visulize the results
import matplotlib.pyplot as plt
df.plot(kind="bar", figsize=(12, 6))
plt.title("Evaluation Metrics for Different Runs")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
