import json
import pysolr

# ----------------------------
# Solr connection
# ----------------------------
SOLR_URL = "http://localhost:8983/solr/trec_covid"
solr = pysolr.Solr(SOLR_URL, always_commit=False, timeout=10)

input_path = "dataset\\corpus_preprocessed.jsonl"

BATCH_SIZE = 500
batch = []

def transform(record):
    return {
        "doc_id": record.get("_id"),
        "title": record.get("title", ""),
        "body": record.get("text", ""),
        "url": record.get("metadata", {}).get("url", ""),
        "pubmed_id": record.get("metadata", {}).get("pubmed_id", "")
    }

print("Starting indexing...")

count = 0

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)

        doc = transform(record)
        batch.append(doc)
        count += 1

        # send batch
        if len(batch) == BATCH_SIZE:
            solr.add(batch)
            print(f"Indexed {count} documents...")
            batch = []

# send remaining docs
if batch:
    solr.add(batch)
    print(f"Indexed final batch. Total: {count}")

# commit changes
solr.commit()

print("Indexing complete!")
print("Total documents indexed:", count)