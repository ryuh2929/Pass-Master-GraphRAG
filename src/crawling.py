import requests
from bs4 import BeautifulSoup
import json
import os
import re
import time
from typing import List, Dict

class ExamCrawler:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        self.output_dir = "data/processed"
        self.image_dir = "data/images"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

    def download_image(self, img_url: str, filename: str) -> str:
        """이미지를 다운로드하고 로컬 경로를 반환합니다."""
        try:
            res = requests.get(img_url, headers=self.headers, timeout=10)
            if res.status_code == 200:
                path = os.path.join(self.image_dir, filename)
                with open(path, 'wb') as f:
                    f.write(res.content)
                return path
        except Exception as e:
            print(f"  ⚠️ 이미지 다운로드 실패: {e}")
        return ""

    def parse_post(self, url: str) -> Dict:
        """블로그 포스트 하나를 파싱하여 문제 리스트를 반환합니다."""
        res = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(res.text, 'lxml')
        
        # 제목에서 정보 추출
        title = soup.find('h1').get_text(strip=True)
        year = re.search(r'(\d{4})년', title).group(1) if re.search(r'(\d{4})년', title) else "Unknown"
        exam_round = re.search(r'(\d)회', title).group(1) if re.search(r'(\d)회', title) else "Unknown"
        
        content = soup.select_one('.entry-content')
        if not content:
            return {}

        problems = []
        
        # 문제 번호(<b>태그)를 기준으로 파싱
        # 블로그마다 구조가 조금씩 다를 수 있어 모든 b 태그를 탐색
        b_tags = content.find_all('b')
        
        for b in b_tags:
            text = b.get_text(strip=True)
            # '1.' 또는 '1) ' 형태의 문제 번호 탐색
            if re.match(r'^\d+[\.\)]', text):
                prob_no = re.sub(r'[^0-9]', '', text.split('.')[0])
                
                # 지문 수집: b태그 이후 다음 b태그가 나오기 전까지의 p, img 태그 수집
                question_text = text
                images = []
                
                curr = b.parent if b.parent.name == 'p' else b
                while curr and curr.next_sibling:
                    curr = curr.next_sibling
                    if curr.name == 'b' or (curr.find and curr.find('b')): break
                    
                    if curr.name == 'p':
                        # 이미지 확인
                        img = curr.find('img')
                        if img:
                            img_src = img.get('src')
                            img_name = f"{year}_{exam_round}_{prob_no}.png"
                            local_path = self.download_image(img_src, img_name)
                            if local_path: images.append(local_path)
                        
                        # 텍스트 추가 (더보기 박스 제외)
                        if 'moreless' not in str(curr.get('class', [])):
                            question_text += "\n" + curr.get_text(strip=True)
                
                # 정답 찾기: 현재 위치 근처의 moreless-content 찾기
                answer = ""
                ans_div = curr.find_parent().find('div', class_='moreless-content') if curr.find_parent() else None
                if not ans_div: # 구조가 다를 경우 주변 탐색
                    ans_div = b.find_all_next('div', class_='moreless-content', limit=1)
                    if ans_div: ans_div = ans_div[0]
                
                if ans_div:
                    answer = ans_div.get_text(strip=True)

                problems.append({
                    "no": prob_no,
                    "question": question_text.strip(),
                    "answer": answer,
                    "images": images
                })

        return {
            "year": year,
            "round": exam_round,
            "url": url,
            "problems": problems
        }

    def run(self, urls: List[str]):
        for url in urls:
            print(f"🔎 파싱 중: {url}")
            data = self.parse_post(url)
            if data and data['problems']:
                filename = f"exam_{data['year']}_{data['round']}.json"
                save_path = os.path.join(self.output_dir, filename)
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"✅ 저장 완료: {save_path} ({len(data['problems'])}문제)")
            time.sleep(1)

if __name__ == "__main__":
    exam_urls = [
        "https://chobopark.tistory.com/540",
        "https://chobopark.tistory.com/554",
        # ... 나머지 URL 리스트
    ]
    crawler = ExamCrawler()
    crawler.run(exam_urls)