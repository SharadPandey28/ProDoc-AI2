# vector_store.py

from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

class LocalSentenceTransformer:
    """
    Adapter that makes SentenceTransformer compatible with LangChain's expectations:
    - callable (so FAISS can call it)
    - embed_documents(list[str]) -> list[list[float]]
    - embed_query(str) -> list[float]
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts):
        """
        Make the object callable. LangChain sometimes calls the embedding object directly.
        Accepts either a single string or a list of strings.
        Returns list-of-lists (embeddings).
        """
        # Normalize input to list
        single_input = False
        if isinstance(texts, str):
            texts = [texts]
            single_input = True

        embs = self.model.encode(texts, convert_to_numpy=False, show_progress_bar=False)
        # SentenceTransformer may return list or numpy - ensure list of lists
        embs_list = [e.tolist() if hasattr(e, "tolist") else e for e in embs]

        return embs_list[0] if single_input else embs_list

    def embed_documents(self, texts):
        """
        LangChain calls this to embed many documents.
        """
        embs = self.model.encode(texts, convert_to_numpy=False, show_progress_bar=False)
        return [e.tolist() if hasattr(e, "tolist") else e for e in embs]

    def embed_query(self, text):
        """
        LangChain calls this to embed a single query.
        Return a single vector (list of floats).
        """
        emb = self.model.encode([text], convert_to_numpy=False, show_progress_bar=False)[0]
        return emb.tolist() if hasattr(emb, "tolist") else emb

def create_vector_store(chunks):
    """
    Build a FAISS vector store from LangChain Document chunks using the local adapter.
    chunks: list of langchain.schema.Document objects (with .page_content and .metadata)
    """
    # create adapter
    local_emb = LocalSentenceTransformer(model_name="all-MiniLM-L6-v2")

    # Use FAISS.from_documents which will call local_emb.embed_documents(...) internally
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=local_emb
    )

    return vector_store
