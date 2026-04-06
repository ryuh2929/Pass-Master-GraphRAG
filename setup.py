import subprocess
from pathlib import Path

def ask(question, options):
    print(f"\n{question}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt['label']}")
    while True:
        choice = input("선택 (번호): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]

def main():
    print("=== Pass-Master-GraphRAG 초기 세팅 ===")

    # 1. 사양 선택
    spec = ask("사양을 선택하세요", [
        {"label": "고사양 (RTX 4070+)  → qwen2.5:14b",  "backend": "ollama",       "model": "qwen2.5:14b"},
        {"label": "중간 (6GB VRAM)     → qwen2.5:7b",   "backend": "ollama",       "model": "qwen2.5:7b"},
        {"label": "저사양 / CPU        → qwen2.5:3b",   "backend": "ollama_light", "model": "qwen2.5:3b"},
        {"label": "OpenAI API 사용",                     "backend": "openai",       "model": None},
    ])

    # 2. .env 생성
    env_path = Path(".env")
    example = Path(".env.example").read_text(encoding="utf-8")

    env_content = example
    env_content = env_content.replace("LLM_BACKEND=", f"LLM_BACKEND={spec['backend']}")

    if spec["backend"] == "openai":
        api_key = input("\nOpenAI API 키를 입력하세요: ").strip()
        env_content = env_content.replace("OPENAI_API_KEY=", f"OPENAI_API_KEY={api_key}")

    neo4j_pw = input("\nNeo4j 비밀번호를 설정하세요: ").strip()
    env_content = env_content.replace("NEO4J_PASSWORD=", f"NEO4J_PASSWORD={neo4j_pw}")

    env_path.write_text(env_content, encoding="utf-8")
    print(f"\n.env 생성 완료")

    # 3. Ollama 모델 다운로드
    if spec["model"]:
        print(f"\n{spec['model']} 다운로드 중... (시간이 걸릴 수 있어요)")
        subprocess.run(["ollama", "pull", spec["model"]], check=True)

    print("\n세팅 완료! 이제 아래 명령어로 실행하세요:")
    print("  docker-compose up -d")
    print("  uv run streamlit run main.py")

if __name__ == "__main__":
    main()