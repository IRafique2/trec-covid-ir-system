import json
import pysolr

# ----------------------------
# Solr connection
# ----------------------------
SOLR_URL = "http://localhost:8983/solr/trec_covid"
solr = pysolr.Solr(SOLR_URL, timeout=10)

queries_path = "dataset/queries.jsonl"
output_path = "output/run_bm25.json"

results = {}

print("Starting BM25 retrieval...")

with open(queries_path, "r", encoding="utf-8") as f:
    for line in f:
        q = json.loads(line)

        query_id = q["_id"]
        query_text = q["text"]

        response = solr.search(
            q=query_text,
            defType="edismax",
            qf="title body",
            rows=1000,
            fl="doc_id,score"
        )

        results[query_id] = [
            (doc["doc_id"], doc["score"])
            for doc in response
        ]

        print(f"Processed query {query_id}, results: {len(response)}")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f)

print("BM25 retrieval complete → saved to", output_path)