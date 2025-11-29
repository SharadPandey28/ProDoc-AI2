# retriever.py

import streamlit as st

@st.cache_resource
def get_retriever(vector_store, k=5):
    """
    Converts FAISS vector store into a retriever.
    
    Parameters:
        vector_store: FAISS vector store instance
        k (int): number of relevant chunks to return

    Returns:
        Retriever object
    """

    if vector_store is None:
        raise ValueError("❌ Vector store is empty. Cannot create retriever.")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    return retriever
