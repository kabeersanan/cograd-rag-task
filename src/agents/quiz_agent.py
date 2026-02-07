from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import LLM_MODEL_NAME, GROQ_API_KEY
from src.agents.prompts import QUIZ_SYSTEM_PROMPT

def generate_quiz(query, context):
    llm = ChatGroq(
        model=LLM_MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=0.3 # Strict for quizzes
    )

    prompt = ChatPromptTemplate.from_template(QUIZ_SYSTEM_PROMPT)
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "query": query,
        "context": context
    })
    return response