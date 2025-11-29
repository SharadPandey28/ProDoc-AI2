# document_loader.py

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

def load_document(file_path: str):
    """
    Loads PDF, DOCX, or TXT file and returns a list of LangChain Document objects.
    """

    # Check file type
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)

    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)

    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)

    else:
        raise ValueError("❌ Unsupported file format. Allowed: PDF, DOCX, TXT")

    # Load and return documents
    return loader.load()
