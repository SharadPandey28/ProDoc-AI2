# rag_chain.py

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough


@st.cache_resource
def build_rag_chain(retriever):
    """
    Builds the RAG chain using OpenAI + a vector retriever.
    Uses API key from Streamlit secrets.
    """

    # ------------------------
    # LLM Model
    # ------------------------
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    # ------------------------
    # Prompt Template
    # ------------------------
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an expert AI assistant.\n\n"
            "Rules:\n"
            "1. If context = 'NO_CONTEXT', summarize the entire document clearly.\n"
            "2. Otherwise, answer ONLY using the given context.\n"
            "3. Keep the answer factual, concise, and well-structured.\n\n"
            "=== Context ===\n"
            "{context}\n\n"
            "=== Question ===\n"
            "{question}\n\n"
            "=== Answer ===\n"
        )
    )

    # ------------------------
    # RAG Chain
    # ------------------------
    chain = (
        RunnableMap({
            "context": lambda x: retriever.get_relevant_documents(x["question"]),
            "question": RunnablePassthrough()
        })
        |
        (
            lambda x: {
                "context": (
                    "\n\n".join(doc.page_content for doc in x["context"])
                    if x["context"] else "NO_CONTEXT"
                ),
                "question": x["question"],
            }
        )
        |
        prompt
        |
        llm
    )

    return chain
