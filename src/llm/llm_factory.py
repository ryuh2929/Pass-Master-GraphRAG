
import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    if provider == "openai":
        # OpenAI 엔진 반환
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        # 로컬 Ollama 엔진 반환
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )