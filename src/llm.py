import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load variables from .env
load_dotenv()


def get_llm():
    """
    Initialize and return the Groq LLM using
    API key and model from .env.
    """

    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Please add it to your .env file."
        )

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.2
    )

    return llm


def test_groq_connection():
    """
    Test Groq connection using credentials from .env.
    """

    try:
        llm = get_llm()

        response = llm.invoke("Say hello")

        return True, "✅ Groq connection successful"

    except Exception as e:
        return False, f"❌ Unable to connect to Groq: {str(e)}"
