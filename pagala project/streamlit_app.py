# rag_chain.py

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough


@st.cache_resource
def build_rag_chain(retriever):
    """
    Builds the RAG chain using OpenAI + FAISS retriever.
    Uses API key from Streamlit secrets.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=st.secrets["OPENAI_API_KEY"]   # 🔥 added secrets integration
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant.\n\n"
            "If the context is 'NO_CONTEXT', summarize the entire document clearly.\n"
            "Otherwise, provide a concise, accurate answer using ONLY the given context.\n\n"
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
