import json
import re

input_path = "dataset\\corpus_20_percent.jsonl"
output_path = "dataset\\corpus_preprocessed.jsonl"

seen_ids = set()
seen_pubmed = set()

removed_empty = 0
removed_duplicates = 0
kept = 0

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\,\;\:\-\(\)]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text: str):
    return text.split()

with open(input_path, "r", encoding="utf-8") as f_in, \
     open(output_path, "w", encoding="utf-8") as f_out:

    for line in f_in:
        record = json.loads(line)

        _id = record.get("_id")
        pubmed_id = record.get("metadata", {}).get("pubmed_id")

        title = record.get("title", "")
        text = record.get("text", "")

        # ---------------------------
        # remove missing/empty fields
        # ---------------------------
        if not isinstance(title, str) or not title.strip():
            removed_empty += 1
            continue

        if not isinstance(text, str) or not text.strip():
            removed_empty += 1
            continue

        # ---------------------------
        # deduplication
        # ---------------------------
        if _id in seen_ids or (pubmed_id and pubmed_id in seen_pubmed):
            removed_duplicates += 1
            continue

        seen_ids.add(_id)
        if pubmed_id:
            seen_pubmed.add(pubmed_id)

        # ---------------------------
        # cleaning
        # ---------------------------
        title_clean = clean_text(title)
        text_clean = clean_text(text)

        # ---------------------------
        # combined field (VERY important for embeddings)
        # ---------------------------
        combined = title_clean + " " + text_clean

        # ---------------------------
        # tokenization (optional feature)
        # ---------------------------
        tokens_title = tokenize(title_clean)
        tokens_text = tokenize(text_clean)
        tokens_combined = tokenize(combined)

        # update record
        record["title"] = title_clean
        record["text"] = text_clean
        record["title_text"] = combined

        record["tokens_title"] = tokens_title
        record["tokens_text"] = tokens_text
        record["tokens"] = tokens_combined

        f_out.write(json.dumps(record) + "\n")
        kept += 1

print("Preprocessing complete!")
print("Kept records:", kept)
print("Removed empty title/text:", removed_empty)
print("Removed duplicates:", removed_duplicates)