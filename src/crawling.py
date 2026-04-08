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
        try:
            res = requests.get(img_url, headers=self.headers, timeout=10)
            if res.status_code == 200:
                path = os.path.join(self.image_dir, filename)
                with open(path, 'wb') as f:
                    f.write(res.content)
                return path
        except Exception: pass
        return ""

    def parse_post(self, url: str) -> Dict:
        res = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(res.text, 'lxml')
        
        # 1. 제목 추출 경로 다각화
        title_text = ""
        # 후보 1: post-cover 클래스 내부의 h1 (보내주신 HTML 구조)
        title_tag = soup.select_one('.post-cover h1')
        # 후보 2: 일반적인 h1
        if not title_tag: title_tag = soup.find('h1')
        # 후보 3: entry-content 내부의 h3 (본문 제목)
        if not title_tag: title_tag = soup.find('h3', string=re.compile(r'복원 문제'))
        
        if title_tag:
            title_text = title_tag.get_text(strip=True)
            print(f"  📌 발견된 제목: {title_text}")

        # 정규식으로 년도와 회차 추출
        year_match = re.search(r'(\d{4})년', title_text)
        round_match = re.search(r'(\d)회', title_text)
        
        year = year_match.group(1) if year_match else "Unknown"
        exam_round = round_match.group(1) if round_match else "Unknown"

        # 제목에서 못 찾으면 URL 번호라도 활용 (덮어쓰기 방지)
        if year == "Unknown":
            post_id = url.split('/')[-1]
            year = f"Post_{post_id}" 

        # 2. 본문 영역 타겟팅
        content = soup.select_one('.entry-content .contents_style') or soup.select_one('.entry-content')
        if not content: return {}

        problems = []
        current_prob = None
        
        # 본문 내의 모든 직계 자식 태그를 순회
        for tag in content.find_all(['p', 'table', 'div', 'figure'], recursive=False):
            text = tag.get_text(strip=True)
            
            # 문제 시작 감지 (예: "1. ", "11. ")
            # p 태그 내부에 b 태그가 있거나, p 태그 자체가 숫자로 시작하는 경우
            is_start = False
            first_text = tag.get_text()
            match = re.match(r'^\s*(\d+)[\.\)]', first_text)
            
            if match:
                # 이전 문제가 있었다면 정답 없이 종료된 것이므로 저장(있을 경우)
                if current_prob:
                    problems.append(current_prob)
                
                prob_no = match.group(1)
                current_prob = {
                    "no": prob_no,
                    "question": first_text.strip(),
                    "answer": "",
                    "images": []
                }
                is_start = True
            
            if current_prob and not is_start:
                # 정답 구역(moreless)을 만났을 때
                if tag.name == 'div' and 'moreLess' in tag.get('class', []):
                    ans_content = tag.select_one('.moreless-content')
                    if ans_content:
                        current_prob["answer"] = ans_content.get_text(strip=True)
                    
                    # 정답을 찾았으므로 한 문제 완성
                    problems.append(current_prob)
                    current_prob = None
                    continue

                # 이미지 추출
                img_tag = tag.find('img')
                if img_tag:
                    img_src = img_tag.get('src') or img_tag.get('data-src')
                    if img_src:
                        img_name = f"{year}_{exam_round}_{current_prob['no']}.png"
                        local_path = self.download_image(img_src, img_name)
                        if local_path: current_prob["images"].append(local_path)

                # 표(table) 또는 일반 텍스트 추가
                if tag.name == 'table':
                    # 표는 텍스트 형태로 변환하여 질문에 추가
                    rows = []
                    for tr in tag.find_all('tr'):
                        cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                        rows.append(" | ".join(cells))
                    current_prob["question"] += "\n[표]\n" + "\n".join(rows)
                else:
                    if text:
                        current_prob["question"] += "\n" + text

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
            if data and data['year'] != "Unknown":
                filename = f"exam_{data['year']}_{data['round']}.json"
                save_path = os.path.join(self.output_dir, filename)
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"✅ 저장 완료: {save_path} ({len(data['problems'])}문제)")
            else:
                print(f"⚠️ 회차 정보를 찾을 수 없어 스킵합니다: {url}")
            time.sleep(1)

if __name__ == "__main__":
    exam_urls = [
        "https://chobopark.tistory.com/554",
        "https://chobopark.tistory.com/540",
        # ... 리스트 생략
    ]
    crawler = ExamCrawler()
    crawler.run(exam_urls)