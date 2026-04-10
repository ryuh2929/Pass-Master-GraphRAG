import json
import os
import sys

def run_integrity_check():
    data_path = "data/processed"
    # exam_으로 시작하는 원본 소스 파일만 필터링
    files = [f for f in os.listdir(data_path) if f.startswith('exam_') and f.endswith('.json')]
    
    if not files:
        print("🔍 검사할 JSON 파일이 없습니다.")
        return

    total_files = len(files)
    passed_files = 0
    failed_reports = []

    print(f"🚀 총 {total_files}개의 시험 데이터 검사를 시작합니다.\n")

    for filename in files:
        file_path = os.path.join(data_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            problems = data if isinstance(data, list) else data.get('problems', [])
            prob_count = len(problems)
            
            # 검증 로직
            issues = []
            if prob_count != 20:
                issues.append(f"문항 수 오류 ({prob_count}개)")
            
            for i, prob in enumerate(problems):
                no = prob.get('no', f"Index_{i}")
                if not str(prob.get('answer', '')).strip():
                    issues.append(f"{no}번 정답 누락")
                if not str(prob.get('question', '')).strip():
                    issues.append(f"{no}번 지문 누락")
                                    
                # 질문의 마지막 부분이 정답과 완전히 동일하게 끝나는지 체크 (중복 방지)
                clean_q = prob.get('question', '').replace("\n", "").replace(" ", "")
                clean_a = prob.get('answer', '').replace("\n", "").replace(" ", "")
                
                if len(clean_a) > 3: # 너무 짧은 정답 제외
                        if clean_a in clean_q[-len(clean_a)-10:]: # 질문 끝에서 정답이 포함되어 있는지 체크
                            issues.append( 
                            f"{no}번 질문 끝에 정답이 중복 포함된 것으로 의심됩니다."
                        )

            if not issues:
                print(f"✅ {filename.ljust(20)} | 통과 (20문항)")
                passed_files += 1
            else:
                issue_str = ", ".join(issues[:3]) + ("..." if len(issues) > 3 else "")
                print(f"❌ {filename.ljust(20)} | 실패 - {issue_str}")
                failed_reports.append((filename, issues))

        except Exception as e:
            print(f"⚠️  {filename.ljust(20)} | 파일 에러: {str(e)}")

    # 최종 결과 요약
    print(f"\n" + "="*50)
    print(f"📊 최종 결과: {passed_files}/{total_files} 통과")
    print("="*50)

    if failed_reports:
        print("\n🔍 미통과 파일 상세 원인:")
        for name, issues in failed_reports:
            print(f"- {name}:")
            for issue in issues:
                print(f"  └ {issue}")
        sys.exit(1) # 실패가 있으면 종료 코드를 1로 설정 (CI/CD 연동용)
    else:
        print("\n🎉 모든 데이터가 완벽합니다!")
        sys.exit(0)

if __name__ == "__main__":
    run_integrity_check()