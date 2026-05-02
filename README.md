# 🔍 Hybrid Search Engine (BM25 + FAISS + Reranker)

This project implements a **production-style hybrid search system** combining:

- 🧠 Lexical search (BM25 via SQLite FTS5)
- 🔎 Semantic search (FAISS + Sentence Transformers)
- 🎯 Neural reranking (CrossEncoder)
- 📊 Evaluation metrics (Recall@K, MRR, latency)
- 💰 Cost estimation simulation
- 💬 Interactive CLI search mode

---

## 🚀 Features

### 1. Hybrid Retrieval
- **BM25 keyword search** using SQLite FTS5
- **Vector search** using FAISS embeddings
- Merged candidate pool with deduplication

### 2. Neural Reranking
- Uses `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Scores query–document pairs
- Produces final ranked results

### 3. Evaluation Pipeline
- Recall@5
- Mean Reciprocal Rank (MRR)
- Latency tracking (ms)
- Ground-truth evaluation set

### 4. Cost Estimation
Simulates per-query cost for:
- Embedding generation
- Reranking compute

### 5. Interactive Mode
Run live queries in CLI or execute full evaluation benchmark.

---

## 🧱 Architecture
