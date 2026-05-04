
import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

def get_llm():
    model_name = os.getenv("LLM_MODEL", "llama3:latest")
    target_temp = 0.1
    
    if "openai" in model_name.lower():
        # OpenAI 엔진 반환
        return ChatOpenAI(
            model="gpt-4o", # 실제 사용할 OpenAI 모델명
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=target_temp 
        )
    else:
        # 로컬 Ollama 엔진 반환
        return ChatOllama(
            model=model_name,
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            temperature=target_temp
        )