from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import LLM_MODEL_NAME, GOOGLE_API_KEY
from src.agents.prompts import ROUTER_SYSTEM_PROMPT

def route_query(query):
    """
    Classifies the user's intent as either 'QUIZ' or 'EXPLAIN'.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is missing.")

    # Architecture Decision: Temperature = 0.0
    # Reasoning: Classification must be deterministic. We don't want 'creativity' here.
    # We want the model to output strictly "QUIZ" or "EXPLAIN" every single time.
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0
    )

    prompt = ChatPromptTemplate.from_template(ROUTER_SYSTEM_PROMPT)
    
    # Architecture Decision: StrOutputParser
    # Reasoning: The output will be a single word string. 
    # This parser strips away all the metadata so we can use a simple if/else statement.
    chain = prompt | llm | StrOutputParser()
    
    print(f"Routing query: '{query}'...")
    intent = chain.invoke({"query": query})
    
    # Architecture Decision: Normalization (.strip().upper())
    # Reasoning: LLMs might sometimes output "Quiz " (with a space) or "quiz" (lowercase).
    # This ensures our if/else logic in main.py never breaks due to formatting.
    return intent.strip().upper()