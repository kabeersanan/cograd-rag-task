from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import LLM_MODEL_NAME, GOOGLE_API_KEY
from src.agents.prompts import QUIZ_SYSTEM_PROMPT

def get_quiz_chain():
    """
    Creates the LangChain sequence for the Quiz Generator agent.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is missing.")

    # Architecture Decision: Temperature = 0.5
    # Reasoning: Quizzes need slightly more creativity than explanations. 
    # We want the 'distractor' options (wrong answers) to be plausible, 
    # which requires a bit more 'imagination' from the model.
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.5
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", QUIZ_SYSTEM_PROMPT),
        ("user", "Context: {context}\n\nGenerate a quiz for topic: {query}")
    ])

    chain = prompt | llm | StrOutputParser()
    return chain

def generate_quiz(query, context):
    """
    Invokes the quiz chain with the specific query and context.
    """
    chain = get_quiz_chain()
    
    response = chain.invoke({
        "context": context,
        "query": query
    })
    
    return response