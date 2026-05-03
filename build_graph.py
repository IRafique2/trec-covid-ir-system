import json
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


# Paths

input_path = r"dataset\\corpus_preprocessed.jsonl"

os.makedirs("output", exist_ok=True)

edges_path = "output/graph_edges.csv"
pagerank_path = "output/pagerank_scores.json"


# Step 4.1 — Load documents

docs = {}
texts = []
doc_ids = []

print("Loading corpus...")

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)

        doc_id = record["_id"]
        title = record.get("title", "")
        body = record.get("text", "")

        combined = (title + " " + body).strip()

        docs[doc_id] = combined
        doc_ids.append(doc_id)
        texts.append(combined)

print("Total documents:", len(docs))


# Step 4.2 — TF-IDF matrix

print("Building TF-IDF matrix...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(texts)

print("TF-IDF shape:", tfidf_matrix.shape)


# Step 4.3 — Similarity computation (batch)

print("Computing cosine similarity...")

similarity_matrix = cosine_similarity(tfidf_matrix)

edges = []

top_k = 5
threshold = 0.15

n = len(doc_ids)

for i in range(n):
    sim_scores = similarity_matrix[i]

    # get top-k indices (excluding self)
    top_indices = np.argsort(sim_scores)[::-1]

    count = 0

    for j in top_indices:
        if i == j:
            continue

        if sim_scores[j] < threshold:
            break

        edges.append((doc_ids[i], doc_ids[j], float(sim_scores[j])))

        count += 1
        if count >= top_k:
            break

print("Total edges:", len(edges))


# Step 4.5 — Build graph

print("Building graph...")

G = nx.DiGraph()

# add nodes
G.add_nodes_from(doc_ids)

# add edges
for src, dst, weight in edges:
    G.add_edge(src, dst, weight=weight)

# save edges
with open(edges_path, "w", encoding="utf-8") as f:
    f.write("source,target,weight\n")
    for src, dst, w in edges:
        f.write(f"{src},{dst},{w}\n")

print("Edges saved to:", edges_path)


# Step 4.6 — PageRank

print("Running PageRank...")

pagerank_scores = nx.pagerank(G, alpha=0.85, max_iter=100)

# save PageRank
with open(pagerank_path, "w", encoding="utf-8") as f:
    json.dump(pagerank_scores, f)

print("PageRank saved to:", pagerank_path)


# Step 4.7 — Graph statistics

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
avg_out_degree = num_edges / num_nodes

top_doc = max(pagerank_scores, key=pagerank_scores.get)
top_score = pagerank_scores[top_doc]

print("\n--- GRAPH STATS ---")
print("Nodes:", num_nodes)
print("Edges:", num_edges)
print("Avg out-degree:", avg_out_degree)
print("Top PageRank score:", top_score)
print("Top document ID:", top_doc)