import requests

SOLR_SCHEMA_URL = "http://localhost:8983/solr/trec_covid/schema"

fields = [
    {"name": "doc_id", "type": "string", "stored": True, "indexed": True},
    {"name": "title", "type": "text_en", "stored": True, "indexed": True},
    {"name": "body", "type": "text_en", "stored": True, "indexed": True},
    {"name": "url", "type": "string", "stored": True, "indexed": False},
    {"name": "pubmed_id", "type": "string", "stored": True, "indexed": False}
]

for field in fields:
    r = requests.post(SOLR_SCHEMA_URL, json={"add-field": field})
    print(field["name"], r.status_code, r.text)