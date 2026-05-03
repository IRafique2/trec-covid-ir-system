import json
import os

os.makedirs("output", exist_ok=True)

# ============================================================
# STEP 1 — Load
# ============================================================
with open("output/run_bm25.json") as f: bm25_raw     = json.load(f)
with open("output/run_lm.json")   as f: lm_raw        = json.load(f)
with open("output/pagerank_scores.json") as f: pagerank_raw = json.load(f)

print(f"BM25: {len(bm25_raw)} queries | LM: {len(lm_raw)} queries | PR: {len(pagerank_raw)} docs")

# ============================================================
# STEP 2 — Normalise
# ============================================================

def minmax_normalize(pairs):
    if not pairs: return {}
    scores = [s for _, s in pairs]
    lo, hi = min(scores), max(scores)
    if hi == lo: return {d: 0.0 for d, _ in pairs}
    return {d: (s - lo) / (hi - lo) for d, s in pairs}

bm25_norm = {qid: minmax_normalize(pairs) for qid, pairs in bm25_raw.items()}
lm_norm   = {qid: minmax_normalize(pairs) for qid, pairs in lm_raw.items()}

pr_vals = list(pagerank_raw.values())
pr_lo, pr_hi = min(pr_vals), max(pr_vals)
pagerank_norm = (
    {k: 1.0 for k in pagerank_raw} if pr_hi == pr_lo
    else {k: (v - pr_lo) / (pr_hi - pr_lo) for k, v in pagerank_raw.items()}
)

# ============================================================
# STEP 3 — Individual runs
# ============================================================

bm25_run = {qid: sorted(n.items(), key=lambda x: x[1], reverse=True)[:1000]
            for qid, n in bm25_norm.items()}

lm_run   = {qid: sorted(n.items(), key=lambda x: x[1], reverse=True)[:1000]
            for qid, n in lm_norm.items()}

pr_run   = {qid: sorted([(d, pagerank_norm.get(d, 0.0)) for d, _ in pairs],
                         key=lambda x: x[1], reverse=True)[:1000]
            for qid, pairs in bm25_raw.items()}

# ============================================================
# STEP 4 — Fused runs
# ============================================================

def linear_fuse(bm25_norm, lm_norm, pagerank_norm, bm25_raw, lm_raw,
                w_bm25, w_lm, w_pr):
    """Weighted linear combination."""
    result = {}
    for qid in bm25_raw.keys():
        q_bm25 = bm25_norm.get(qid, {})
        q_lm   = lm_norm.get(qid,   {})
        cands  = set(q_bm25) | set(q_lm)
        scores = {
            d: w_bm25 * q_bm25.get(d, 0.0)
             + w_lm   * q_lm.get(d,   0.0)
             + w_pr   * pagerank_norm.get(d, 0.0)
            for d in cands
        }
        result[qid] = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1000]
    return result


def rrf_fuse(bm25_norm, lm_norm, pagerank_norm, bm25_raw, k=60):
    """Reciprocal Rank Fusion."""
    result = {}
    for qid in bm25_raw.keys():
        bm25_ranked = sorted(bm25_norm.get(qid, {}).items(),
                             key=lambda x: x[1], reverse=True)
        lm_ranked   = sorted(lm_norm.get(qid,   {}).items(),
                             key=lambda x: x[1], reverse=True)
        pr_ranked   = sorted([(d, pagerank_norm.get(d, 0.0))
                               for d, _ in bm25_raw.get(qid, [])],
                             key=lambda x: x[1], reverse=True)

        bm25_rank = {d: r for r, (d, _) in enumerate(bm25_ranked, 1)}
        lm_rank   = {d: r for r, (d, _) in enumerate(lm_ranked,   1)}
        pr_rank   = {d: r for r, (d, _) in enumerate(pr_ranked,   1)}

        cands   = set(bm25_rank) | set(lm_rank)
        penalty = len(cands) + 1

        scores = {
            d: 1/(k + bm25_rank.get(d, penalty))
             + 1/(k + lm_rank.get(d,   penalty))
             + 1/(k + pr_rank.get(d,   penalty))
            for d in cands
        }
        result[qid] = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1000]
    return result


# Four fusion variants to compare
fused_runs = {
    # BM25-dominant: give LM weight only if it's better than before
    "FUSED_BM25_DOM":  linear_fuse(bm25_norm, lm_norm, pagerank_norm,
                                   bm25_raw, lm_raw,
                                   w_bm25=0.70, w_lm=0.28, w_pr=0.02),

    # Equal BM25+LM, ignore PR
    "FUSED_NO_PR":     linear_fuse(bm25_norm, lm_norm, pagerank_norm,
                                   bm25_raw, lm_raw,
                                   w_bm25=0.50, w_lm=0.50, w_pr=0.00),

    # RRF with k=60 (standard)
    "FUSED_RRF_60":    rrf_fuse(bm25_norm, lm_norm, pagerank_norm,
                                bm25_raw, k=60),

    # RRF with k=10 (amplifies rank differences — rewards consistent top-ranks)
    "FUSED_RRF_10":    rrf_fuse(bm25_norm, lm_norm, pagerank_norm,
                                bm25_raw, k=10),
}

# ============================================================
# STEP 5 — Write all TREC files
# ============================================================

def write_trec(run_dict, filepath, tag):
    with open(filepath, "w") as f:
        for qid, docs in run_dict.items():
            for rank, (doc_id, score) in enumerate(docs[:1000], 1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.8f} {tag}\n")
    print(f"  Written: {filepath}")

print("\nWriting TREC files...")
write_trec(bm25_run,  "output/run_bm25.trec",    "BM25")
write_trec(lm_run,    "output/run_lm.trec",       "LM")
write_trec(pr_run,    "output/run_pagerank.trec", "PAGERANK")

for tag, run in fused_runs.items():
    write_trec(run, f"output/run_{tag.lower()}.trec", tag)

# Keep a canonical "run_fused.trec" pointing to best expected variant
write_trec(fused_runs["FUSED_BM25_DOM"], "output/run_fused.trec", "FUSED")

print("\nDone.")