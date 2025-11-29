# document_loader.py

import tempfile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader


def load_document_from_streamlit(uploaded_file):
    """
    Loads a PDF, DOCX, or TXT uploaded via Streamlit's st.file_uploader().
    Saves file temporarily and returns LangChain Document objects.
    """

    if uploaded_file is None:
        raise ValueError("❌ No file uploaded.")

    file_name = uploaded_file.name.lower()

    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_name) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Select the correct loader
    if file_name.endswith(".pdf"):
        loader = PyPDFLoader(temp_path)

    elif file_name.endswith(".docx"):
        loader = Docx2txtLoader(temp_path)

    elif file_name.endswith(".txt"):
        loader = TextLoader(temp_path)

    else:
        raise ValueError("❌ Unsupported file format. Allowed: PDF, DOCX, TXT")

    return loader.load()


def load_document(file_path: str):
    """
    Original loader for local file paths.
    Works for local testing or CLI use.
    """

    file_name = file_path.lower()

    if file_name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)

    elif file_name.endswith(".docx"):
        loader = Docx2txtLoader(file_path)

    elif file_name.endswith(".txt"):
        loader = TextLoader(file_path)

    else:
        raise ValueError("❌ Unsupported file format. Allowed: PDF, DOCX, TXT")

    return loader.load()
