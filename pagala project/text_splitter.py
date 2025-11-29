# text_splitter.py

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter


@st.cache_resource
def get_text_splitter():
    """
    Returns a cached text splitter instance for efficient reuse.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=800,        # Size of each chunk
        chunk_overlap=150,     # Overlap to preserve context
        separators=["\n\n", "\n", ".", " ", ""]
    )


def split_documents(documents):
    """
    Splits loaded documents into smaller chunks suitable for RAG.
    Returns a list of LangChain Document objects (chunks).
    """
    splitter = get_text_splitter()
    return splitter.split_documents(documents)
