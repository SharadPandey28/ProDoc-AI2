# profession_chain.py

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

def build_profession_chain():

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3
    )

    prompt = PromptTemplate(
        input_variables=["profession", "rag_answer"],
        template=(
            "You are an expert {profession}.\n"
            "Based on the document answer below, write a conclusion from the "
            "{profession}'s perspective with key insights and actionable points.\n\n"
            "Document Answer:\n{rag_answer}\n\n"
            "Conclusion:"
        )
    )

    # Build chain using pipeline syntax
    chain = prompt | llm

    return chain
