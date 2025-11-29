# embeddings.py

from sentence_transformers import SentenceTransformer

def get_embeddings():
    """
    Returns a local embedding model (no API, no quota, free).
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model
