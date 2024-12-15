def query_index(index, query, model, top_k=5):
    """Queries the FAISS index."""
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices
