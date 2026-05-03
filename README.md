# Hybrid Information Retrieval System for TREC-COVID

##  Overview

This project implements a **hybrid information retrieval (IR) system** for the TREC-COVID dataset, combining multiple ranking paradigms:

* **Lexical retrieval (BM25 via Apache Solr)**
* **Probabilistic language modeling (Dirichlet-smoothed query likelihood)**
* **Graph-based ranking (PageRank on a document similarity graph)**
* **Score fusion techniques (linear fusion + RRF variants)**

The goal is to evaluate how different retrieval strategies and their combinations affect document ranking quality in a biomedical search setting.

---

## ⚙️ System Architecture

The pipeline consists of five major components:

1. **Indexing Engine (Apache Solr)**

   * BM25-based lexical retrieval
   * Full-text indexing of ~171K COVID-19 scientific papers

2. **Graph Construction Module**

   * TF-IDF vectorization of documents
   * Cosine similarity-based document graph
   * PageRank computation for global importance scoring

3. **Language Model Retriever**

   * Dirichlet-smoothed query likelihood model
   * Re-ranking of BM25 candidate documents

4. **Fusion Engine**

   * Linear score combination
   * Reciprocal Rank Fusion (RRF)

5. **Evaluation Framework**

   * TREC evaluation metrics via `pytrec_eval`
   * MAP, nDCG@10/20, Precision@10/20

---

##  Dataset

* **Corpus:** TREC-COVID scientific document collection (~171,000 papers)
* **Queries:** 50 biomedical search queries
* **Relevance judgments:** Graded (0 = not relevant, 1 = partially relevant, 2 = highly relevant)

---

##  Methods

### 1. BM25 Retrieval (Baseline)

Implemented using Apache Solr with default BM25 settings:

* k1 = 1.2
* b = 0.75

---

### 2. Language Model (Dirichlet Smoothing)

A query likelihood model:

* Smoothing parameter: μ = 2000
* Uses term frequency statistics for probabilistic ranking
* Applied over top-1000 BM25 candidates

---

### 3. Graph-Based Ranking (PageRank)

Since TREC-COVID lacks citation links:

* A **document similarity graph** is constructed using TF-IDF cosine similarity
* Edge creation rule:

  * Top-5 nearest neighbors per document
  * Similarity threshold ≥ 0.15
* PageRank computed with:

  * damping factor α = 0.85

---

### 4. Score Fusion Strategies

Multiple fusion approaches are evaluated:

* Linear weighted fusion:

  ```
  Score = 0.35·BM25 + 0.35·LM + 0.30·PageRank
  ```

* Reciprocal Rank Fusion (RRF):

  * k = 10, 60 variants tested

* Ablation study:

  * Fusion with and without PageRank

---

##  Results

| Model              | MAP        | nDCG@10    | nDCG@20    | P@10      | P@20      |
| ------------------ | ---------- | ---------- | ---------- | --------- | --------- |
| BM25               | 0.0155     | 0.3696     | 0.3358     | 0.418     | 0.376     |
| LM (Dirichlet)     | 0.0182     | 0.4512     | 0.4031     | 0.506     | 0.444     |
| PageRank           | 0.0047     | 0.0838     | 0.0800     | 0.100     | 0.097     |
| Fusion (BM25+LM)   | **0.0187** | **0.4551** | **0.4091** | **0.518** | **0.457** |
| Fusion (+PageRank) | 0.0175     | 0.4269     | 0.3843     | 0.486     | 0.427     |
| RRF (k=60)         | 0.0172     | 0.4540     | 0.4079     | 0.512     | 0.449     |
| RRF (k=10)         | 0.0156     | 0.4430     | 0.3716     | 0.494     | 0.392     |

---

##  Key Findings

* **Language Model retrieval consistently outperforms BM25**, especially in top-ranked results.
* **Score fusion improves performance**, with BM25 + LM achieving the best overall results.
* **PageRank underperforms significantly**, likely due to:

  * Synthetic construction of the similarity graph
  * Lack of true citation or hyperlink structure
* Graph-based signals provide limited benefit in this dataset compared to lexical and probabilistic methods.

---

##  Tech Stack

* **Apache Solr 9** (Indexing + BM25 retrieval)
* **Python 3.10**
* `pysolr`
* `scikit-learn`
* `networkx`
* `numpy`, `pandas`
* `pytrec_eval`
* `tqdm`

---

##  Project Structure

```

│
├── data/                 # TREC-COVID dataset
├──                  # Source code
│   ├── index.py
│   ├── build_graph.py
│   ├── retrieve_bm25.py
│   ├── retrieve_lm.py
│   ├── fuse_scores.py
│   └── evaluate.py
│
├── outputs/              # Run files and results
├── evaluation/           # Metric outputs
├── README.md

```

---

##  How to Run

1. Start Apache Solr (via Docker)
2. Create collection `trec_covid`
3. Run indexing:

   ```bash
   python src/index.py
   ```
4. Build similarity graph:

   ```bash
   python src/build_graph.py
   ```
5. Run retrieval models:

   ```bash
   python src/retrieve_bm25.py
   python src/retrieve_lm.py
   ```
6. Fuse results:

   ```bash
   python src/fuse_scores.py
   ```
7. Evaluate:

   ```bash
   python src/evaluate.py
   ```

---

##  Conclusion

This project demonstrates a full-scale information retrieval pipeline integrating lexical, probabilistic, and graph-based ranking methods. Experimental results show that **language model-based retrieval and score fusion significantly improve ranking effectiveness**, while graph-based PageRank requires more realistic link structures to be effective in biomedical IR tasks.

---

## 📄 License

This project is intended for educational and academic use.



