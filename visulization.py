import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
edges_df = pd.read_csv("output/graph_edges.csv")

with open("output/pagerank_scores.json", "r") as f:
    pagerank = json.load(f)

# -----------------------------
# Build full graph
# -----------------------------
G = nx.DiGraph()

for _, row in edges_df.iterrows():
    G.add_edge(row["source"], row["target"], weight=row["weight"])

# -----------------------------
# Select TOP 6 documents
# -----------------------------
top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:6]
nodes = [n for n, _ in top_nodes]

# -----------------------------
# Create tiny subgraph
# -----------------------------
subgraph = G.subgraph(nodes).copy()

# -----------------------------
# Remove very weak edges
# -----------------------------
threshold = 0.2

edges_to_keep = [
    (u, v)
    for u, v, d in subgraph.edges(data=True)
    if d["weight"] >= threshold
]

clean_graph = subgraph.edge_subgraph(edges_to_keep).copy()

# -----------------------------
# Layout (simple & readable)
# -----------------------------
pos = nx.spring_layout(clean_graph, seed=42)

# -----------------------------
# Node size by PageRank
# -----------------------------
node_sizes = [
    pagerank.get(n, 0) * 80000
    for n in clean_graph.nodes()
]

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8, 6))

nx.draw_networkx_nodes(
    clean_graph,
    pos,
    node_size=node_sizes,
    alpha=0.9
)

nx.draw_networkx_edges(
    clean_graph,
    pos,
    width=1.2,
    alpha=0.6
)

nx.draw_networkx_labels(
    clean_graph,
    pos,
    font_size=10
)

plt.title("Top 6 Document Similarity Graph")
plt.axis("off")
plt.show()