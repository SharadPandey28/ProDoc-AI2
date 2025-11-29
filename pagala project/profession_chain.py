# profession_chain.py

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

@st.cache_resource
def build_profession_chain():
    """
    Builds the profession-based conclusion generator chain.
    Uses OpenAI API key from Streamlit secrets.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=st.secrets["OPENAI_API_KEY"]   # <-- important change
    )

    prompt = PromptTemplate(
        input_variables=["profession", "rag_answer"],
        template=(
            "You are an expert {profession}.\n"
            "Your job is to write a clear, professional-quality conclusion "
            "from the perspective of a {profession} based on the analysis below.\n\n"
            "Analysis / Document Answer:\n{rag_answer}\n\n"
            "Write a concise, meaningful, and practical conclusion:"
        )
    )

    return prompt | llm
