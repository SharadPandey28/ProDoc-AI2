# rag_chain.py

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

def build_rag_chain(retriever):

    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.2
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
       template=(
         "If context is 'NO_CONTEXT', summarize the entire document.\n"
        "Otherwise, use the context to answer.\n\n"
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
        (lambda x: {
        "context": (
        "\n\n".join(doc.page_content for doc in x["context"])
        if x["context"] else "NO_CONTEXT"
        ),
        "question": x["question"]
        })

        |
        prompt
        |
        llm
    )

    return chain
