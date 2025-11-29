# streamlit_app.py

import streamlit as st
import tempfile

# Import your modules
from document_loader import load_document_from_streamlit
from text_splitter import split_documents
from embeddings import get_embeddings
from vector_store import create_vector_store
from retriever import get_retriever
from rag_chain import build_rag_chain
from profession_chain import build_profession_chain


# --- Streamlit Page Config ---
st.set_page_config(page_title="RAG Document Analyzer", layout="wide")

st.title("📄 RAG Document Analyzer with Profession-Based Conclusion")
st.write("Upload a document → Ask a question → Get a profession-specific conclusion.")


# -------------------------------
# 1️⃣ File Upload Section
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "docx", "txt"]
)

# Stop early until a file is uploaded
if uploaded_file is None:
    st.info("👆 Please upload a PDF, DOCX, or TXT file to begin.")
    st.stop()


# -------------------------------
# 2️⃣ Load & Process Document
# -------------------------------
with st.spinner("📥 Loading document..."):
    try:
        docs = load_document_from_streamlit(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        st.stop()

with st.spinner("✂️ Splitting into chunks..."):
    chunks = split_documents(docs)

with st.spinner("🧠 Creating vector store..."):
    vector_store = create_vector_store(chunks)

retriever = get_retriever(vector_store)

st.success("✅ Document processed successfully!")


# -------------------------------
# 3️⃣ User Inputs
# -------------------------------
question = st.text_input("Enter your question:")

profession = st.selectbox(
    "Select your profession for customized conclusion:",
    [
        "Engineer", "Doctor", "Lawyer", "Student",
        "Business Manager", "Researcher", "Teacher", "Developer"
    ]
)


# -------------------------------
# 4️⃣ Perform RAG + Profession Conclusion
# -------------------------------
if st.button("Generate Answer"):

    if question.strip() == "":
        st.warning("⚠ Please enter a valid question.")
        st.stop()

    with st.spinner("🔍 Generating RAG answer..."):
        rag_chain = build_rag_chain(retriever)
        rag_response = rag_chain.invoke({"question": question})

    with st.spinner("🧑‍🏫 Creating profession-based conclusion..."):
        profession_chain = build_profession_chain()
        final_conclusion = profession_chain.invoke({
            "profession": profession,
            "rag_answer": rag_response.content
        })

    # -------------------------------
    # 5️⃣ Display Results
    # -------------------------------
    st.subheader("📘 RAG Answer")
    st.write(rag_response.content)

    st.subheader(f"🎯 Conclusion ({profession} Perspective)")
    st.write(final_conclusion.content)
