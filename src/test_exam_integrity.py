import unittest
import json
import os

class TestExamData(unittest.TestCase):
    def setUp(self):
        # 크롤링된 결과가 저장된 경로
        self.data_path = "data/processed"
        self.files = [f for f in os.listdir(self.data_path) if f.startswith('exam_') and f.endswith('.json')]

    def test_exam_integrity(self):
        """전체 시험 데이터 파일의 무결성을 검증합니다."""
        for filename in self.files:
            with self.subTest(filename=filename):
                file_full_path = os.path.join(self.data_path, filename)
                with open(file_full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 데이터 타입에 따른 처리
                if isinstance(data, list):
                    problems = data
                else:
                    problems = data.get('problems', [])

                # 1. 문항 수 체크 (반드시 20문제여야 함)
                self.assertEqual(len(problems), 20, f"❌ {filename}: 문항 수가 {len(problems)}개입니다. (20개 기대)")

                for i, prob in enumerate(problems):
                    # 문항 번호가 꼬였을 수도 있으니 index 활용
                    prob_no = prob.get('no', f"인덱스_{i}")
                    
                    # 2. 정답 존재 여부 체크 (공백, \n, 단순 띄어쓰기 포함 방지)
                    answer = prob.get('answer', '')
                    self.assertTrue(
                        bool(answer and str(answer).strip()), 
                        f"❌ {filename} [{prob_no}번]: 정답이 비어있거나 공백입니다."
                    )

                    # 3. 질문 내용 존재 여부 체크
                    question = prob.get('question', '')
                    self.assertTrue(
                        bool(question and question.strip()), 
                        f"❌ {filename} [{prob_no}번]: 질문 내용이 없습니다."
                    )
                    
                    # 4. 질문과 정답의 중복 체크
                    # 질문의 마지막 부분이 정답과 완전히 동일하게 끝나는지 체크 (중복 방지)
                    clean_q = question.replace("\n", "").replace(" ", "")
                    clean_a = answer.replace("\n", "").replace(" ", "")
                    
                    # 정답이 질문에 포함되어 있되, 질문의 핵심 내용보다 긴 경우는 오류로 간주 (경험적 수치 0.8)
                    if len(clean_a) > 5: # 너무 짧은 정답 제외
                         self.assertFalse(
                             clean_a in clean_q[-len(clean_a)-10:], 
                             f"❌ {filename} [{prob_no}번]: 질문 끝에 정답이 중복 포함된 것으로 의심됩니다."
                         )

if __name__ == "__main__":
    # 테스트 실행
    unittest.main()