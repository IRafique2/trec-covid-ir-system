import json
import math
import re
import pysolr
from collections import Counter

# -------------------------------------------------------
# Requires:  pip install pysolr nltk
# First run: python -c "import nltk; nltk.download('stopwords')"
# -------------------------------------------------------
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

SOLR_URL     = "http://localhost:8983/solr/trec_covid"
solr         = pysolr.Solr(SOLR_URL, timeout=60)
queries_path = "dataset/queries.jsonl"
output_path  = "output/run_lm.json"

MU        = 2000    # Dirichlet smoothing — tunable
PAGE_SIZE = 500

# ============================================================
# Text normalisation pipeline
# ============================================================
# Solr's default StandardAnalyzer does:
#   1. Lowercase
#   2. Tokenise on whitespace / punctuation
#   3. Remove stop words (English)
#   4. (sometimes) Porter stemming
#
# We mirror this so that tf counts from stored text and cf counts
# from the corpus scan use the same vocabulary.
# ============================================================

_stopwords = set(stopwords.words("english"))
_stemmer   = PorterStemmer()
_tok_re    = re.compile(r"[a-z0-9]+")   # alphanumeric tokens only


def analyse(text):
    """
    Lowercase → tokenise on non-alphanumeric → remove stopwords → stem.
    Returns a list of processed tokens.
    """
    if not text:
        return []
    text   = text.lower()
    tokens = _tok_re.findall(text)
    return [_stemmer.stem(t) for t in tokens if t not in _stopwords and len(t) > 1]


# ============================================================
# STEP 1 — Build corpus statistics from stored text
# ============================================================

print("Building corpus statistics (scanning all documents)...")

global_term_freq = Counter()
total_tokens     = 0
total_docs       = 0
cursor_mark      = "*"

while True:
    res = solr.search(
        q="*:*",
        rows=PAGE_SIZE,
        fl="doc_id,title,body",
        sort="id asc",
        cursorMark=cursor_mark,
    )
    if not res.docs:
        break

    for doc in res.docs:
        title = doc.get("title", "") or ""
        body  = doc.get("body",  "") or ""
        if isinstance(title, list): title = " ".join(title)
        if isinstance(body,  list): body  = " ".join(body)

        tokens = analyse(title + " " + body)
        global_term_freq.update(tokens)
        total_tokens += len(tokens)
        total_docs   += 1

    next_cursor = res.raw_response.get("nextCursorMark", cursor_mark)
    if next_cursor == cursor_mark:
        break
    cursor_mark = next_cursor

    if total_docs % 5000 == 0:
        print(f"  Scanned {total_docs:,} docs | tokens: {total_tokens:,}")

print(f"\nCorpus scan complete:")
print(f"  Documents       : {total_docs:,}")
print(f"  Vocabulary      : {len(global_term_freq):,}")
print(f"  Total tokens C  : {total_tokens:,}")
print(f"  Avg doc length  : {total_tokens/total_docs:.1f}")

if total_tokens == 0:
    raise RuntimeError("total_tokens is 0 — check title/body fields in Solr.")


# ============================================================
# STEP 2 — LM retrieval
# ============================================================

print("\nStarting LM retrieval...")
lm_results = {}

with open(queries_path, "r", encoding="utf-8") as qf:
    for line in qf:
        q = json.loads(line)

        query_id    = q["_id"]
        query_text  = q["text"]
        query_terms = analyse(query_text)   # same pipeline as corpus

        if not query_terms:
            lm_results[query_id] = []
            continue

        # BM25 candidate retrieval — fetch stored text in same call
        bm25_res = solr.search(
            q=query_text,
            defType="edismax",
            qf="title body",
            rows=1000,
            fl="doc_id,title,body,score"
        )

        scores = {}

        for doc in bm25_res:
            doc_id = doc["doc_id"]

            title = doc.get("title", "") or ""
            body  = doc.get("body",  "") or ""
            if isinstance(title, list): title = " ".join(title)
            if isinstance(body,  list): body  = " ".join(body)

            tokens     = analyse(title + " " + body)
            tf_map     = Counter(tokens)
            doc_length = len(tokens)

            if doc_length == 0:
                scores[doc_id] = float("-inf")
                continue

            # Dirichlet-smoothed unigram LM
            score = 0.0
            for term in query_terms:
                tf           = tf_map.get(term, 0)
                cf           = global_term_freq.get(term, 1)
                p_collection = cf / total_tokens
                p_doc        = (tf + MU * p_collection) / (doc_length + MU)
                score       += math.log(p_doc + 1e-300)

            scores[doc_id] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        lm_results[query_id] = ranked[:1000]

        valid  = [s for _, s in ranked if s > float("-inf")]
        rng    = max(valid) - min(valid) if valid else 0
        unique = len(set(round(s, 4) for s in valid))
        top3   = [round(s, 4) for _, s in ranked[:3]]
        print(f"  Query {query_id}: range={rng:.4f} | unique={unique} | top3={top3}")

# ============================================================
# STEP 3 — Save
# ============================================================
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(lm_results, f)

print(f"\nLM retrieval complete → {output_path}")