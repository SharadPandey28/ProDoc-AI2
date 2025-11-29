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

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=st.secrets["OPENAI_API_KEY"]   # <-- added
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an expert assistant.\n"
            "If context is 'NO_CONTEXT', summarize the entire document clearly and concisely.\n"
            "Otherwise, answer the question strictly based on the provided context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
    )

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
                "question": x["question"]
            }
        )
        |
        prompt
        |
        llm
    )

    return chain
