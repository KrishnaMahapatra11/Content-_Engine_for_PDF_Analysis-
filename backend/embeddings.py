from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def generate_embeddings(content):
    """Generates embeddings for the given text."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(content)
    return embeddings

def initialize_faiss(embedding_dim):
    """Initializes a FAISS index."""
    return faiss.IndexFlatL2(embedding_dim)

def add_to_faiss(index, embeddings):
    """Adds embeddings to the FAISS index."""
    index.add(np.array(embeddings, dtype="float32"))
    return index
