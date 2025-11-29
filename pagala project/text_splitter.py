# text_splitter.py

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents):
    """
    Splits the loaded documents into smaller chunks for RAG.
    Returns a list of chunked LangChain Document objects.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # size of each chunk
        chunk_overlap=150,    # overlapping words to preserve context
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    return chunks
