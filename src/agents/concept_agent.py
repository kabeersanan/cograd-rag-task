from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import LLM_MODEL_NAME, GROQ_API_KEY
from src.agents.prompts import CONCEPT_SYSTEM_PROMPT

def generate_explanation(query, context, history):
    llm = ChatGroq(
        model=LLM_MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=0.5 # Slight creativity for explanations
    )

    prompt = ChatPromptTemplate.from_template(CONCEPT_SYSTEM_PROMPT)
    chain = prompt | llm | StrOutputParser()

    history_str = "\n".join([f"{role}: {msg}" for role, msg in history])

    response = chain.invoke({
        "query": query,
        "context": context,
        "history": history_str
    })
    return response