from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from config import (
    LLM_PROVIDER, 
    GOOGLE_API_KEY, 
    OPENAI_API_KEY, 
    GEMINI_MODEL, 
    OPENAI_MODEL
)
from logger import setup_logger

logger = setup_logger("LLMFactory")

def get_llm(temperature=0.7):
    """
    Returns an LLM instance based on the LLM_PROVIDER setting in config.py.
    """
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not found in environment!")
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'")
        
        logger.info(f"Initializing OpenAI LLM: {OPENAI_MODEL}")
        return ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=temperature
        )
    
    # Default to Gemini
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not found in environment!")
        raise ValueError("GOOGLE_API_KEY is required when LLM_PROVIDER is 'gemini'")
        
    logger.info(f"Initializing Gemini LLM: {GEMINI_MODEL}")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature
    )
