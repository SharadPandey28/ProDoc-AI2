# embeddings.py

import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource
def get_embeddings():
    """
    Returns a cached local embedding model (fast + efficient).
    No API keys required.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model
