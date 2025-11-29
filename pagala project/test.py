from document_loader import load_document
from text_splitter import split_documents
from vector_store import create_vector_store
from retriever import get_retriever

docs = load_document("sample.pdf")
chunks = split_documents(docs)

vs = create_vector_store(chunks)
retriever = get_retriever(vs, k=3)

res = retriever.get_relevant_documents("What is the main topic?")
print("Got", len(res), "results. First result:", res[0].page_content[:300])
