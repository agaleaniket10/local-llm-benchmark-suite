import sqlite3
import numpy as np
import time
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

# =========================================================
# MODELS
# =========================================================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# =========================================================
# DATASET (docs + ground truth queries)
# =========================================================
docs = [
    "Invoice 12345 payment pending",
    "Refund policy allows 30 days return",
    "Customer account locked due to suspicious activity",
    "FAISS is a vector similarity search library",
    "Hybrid search combines lexical and semantic retrieval",
]

# ground truth for evaluation (what should be relevant)
eval_set = [
    {"query": "invoice payment", "relevant": ["Invoice 12345 payment pending"]},
    {"query": "refund policy", "relevant": ["Refund policy allows 30 days return"]},
    {
        "query": "vector search library",
        "relevant": ["FAISS is a vector similarity search library"],
    },
]

# =========================================================
# BM25 (SQLite FTS5)
# =========================================================
conn = sqlite3.connect(":memory:")
c = conn.cursor()
c.execute("CREATE VIRTUAL TABLE docs USING fts5(content)")

for d in docs:
    c.execute("INSERT INTO docs(content) VALUES (?)", (d,))
conn.commit()


def bm25_search(query, k=5):
    # Sanitize query for FTS5: keep only alphanumeric and spaces
    safe_query = " ".join(word for word in query.split() if word.isalnum())
    if not safe_query:
        return []
    try:
        res = c.execute("SELECT content FROM docs WHERE docs MATCH ?", (safe_query,))
        return [r[0] for r in res.fetchall()][:k]
    except sqlite3.OperationalError:
        return []


# =========================================================
# VECTOR SEARCH (FAISS)
# =========================================================
doc_embeddings = embed_model.encode(docs)
dim = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dim)
index.add(np.array(doc_embeddings).astype("float32"))


def vector_search(query, k=5):
    q = embed_model.encode([query]).astype("float32")
    _, idx = index.search(q, k)
    return [docs[i] for i in idx[0]]


# =========================================================
# HYBRID + RERANK
# =========================================================
def rerank(query, docs):
    pairs = [(query, d) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked]


def hybrid_search(query):
    bm = bm25_search(query)
    vec = vector_search(query)

    merged = list(set(bm + vec))
    return rerank(query, merged)


# =========================================================
# METRICS
# =========================================================
def recall_at_k(predicted, relevant, k=5):
    pred_k = predicted[:k]
    return int(any(r in pred_k for r in relevant))


def mrr(predicted, relevant):
    for i, p in enumerate(predicted):
        if p in relevant:
            return 1 / (i + 1)
    return 0


# =========================================================
# COST ESTIMATOR
# =========================================================
def estimate_cost(num_queries):
    # fake realistic approximation
    embedding_cost = num_queries * 0.0001
    rerank_cost = num_queries * 0.0003
    return embedding_cost + rerank_cost


# =========================================================
# BENCHMARK PIPELINE
# =========================================================
def evaluate_system():
    print("\n=== EVALUATION START ===")

    total_recall = 0
    total_mrr = 0
    latencies = []

    for item in eval_set:
        start = time.time()

        results = hybrid_search(item["query"])

        latency = (time.time() - start) * 1000
        latencies.append(latency)

        r = recall_at_k(results, item["relevant"])
        m = mrr(results, item["relevant"])

        total_recall += r
        total_mrr += m

        print(f"\nQuery: {item['query']}")
        print("Results:", results)
        print("Recall@5:", r)
        print("MRR:", round(m, 3))
        print("Latency(ms):", round(latency, 2))

    print("\n=== FINAL METRICS ===")
    print("Avg Recall@5:", round(total_recall / len(eval_set), 3))
    print("Avg MRR:", round(total_mrr / len(eval_set), 3))
    print("Avg Latency(ms):", round(sum(latencies) / len(latencies), 2))

    print("\nEstimated cost for 1000 queries: $", round(estimate_cost(1000), 4))


# =========================================================
# INTERACTIVE MODE
# =========================================================
def interactive():
    while True:
        q = input("\nEnter query (or 'eval'): ")

        if q == "exit":
            break
        elif q == "eval":
            evaluate_system()
            continue

        start = time.time()
        results = hybrid_search(q)
        latency = (time.time() - start) * 1000

        print("\nTop results:")
        for r in results:
            print("-", r)

        print("Latency(ms):", round(latency, 2))


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    interactive()
