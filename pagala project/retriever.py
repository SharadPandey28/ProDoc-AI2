# retriever.py

def get_retriever(vector_store, k=5):
    """
    Converts FAISS vector store into a retriever.
    
    k = number of most relevant chunks to return
    """

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    return retriever
