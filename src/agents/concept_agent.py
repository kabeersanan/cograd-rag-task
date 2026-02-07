from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import LLM_MODEL_NAME, GOOGLE_API_KEY
from src.agents.prompts import CONCEPT_SYSTEM_PROMPT

def get_concept_chain():
    """
    Creates the LangChain sequence for the Concept Explainer agent.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is missing.")

    # Architecture Decision: Temperature = 0.3
    # Reasoning: We want the explanation to be factual (low randomness) but 
    # slightly creative enough to generate good analogies. 0.3 is a safe balance.
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    # Architecture Decision: ChatPromptTemplate
    # Reasoning: This binds our "Teacher Persona" (system prompt) with the 
    # specific user input (query) and the textbook data (context).
    prompt = ChatPromptTemplate.from_messages([
        ("system", CONCEPT_SYSTEM_PROMPT),
        ("user", "Context: {context}\n\nQuestion: {query}")
    ])

    # Architecture Decision: StrOutputParser
    # Reasoning: The LLM returns a complex object (AIMessage). We only want 
    # the actual text string to show the student. This parser extracts it automatically.
    chain = prompt | llm | StrOutputParser()
    return chain

def generate_explanation(query, context):
    """
    Invokes the concept chain with the specific query and context.
    """
    chain = get_concept_chain()
    
    response = chain.invoke({
        "context": context,
        "query": query
    })
    
    return response