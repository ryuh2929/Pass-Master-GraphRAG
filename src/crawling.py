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
        
        # 모든 p, table, div를 순서대로 탐색
        # 단, 중첩된 구조를 피하기 위해 content 바로 아래의 요소들 위주로 탐색 로직 변경
        elements = content.find_all(['p', 'table', 'div', 'figure'])
        
        for tag in elements:
            # 이미 처리된 태그(예: moreless 내부의 p)는 스킵하기 위한 필터링
            if tag.find_parent('div', class_='moreless-content'):
                continue

            text = tag.get_text(strip=True)
            
            # [패턴 강화] <b>1.</b> 또는 <span>1.</span> 등 숫자 시작 패턴 탐색
            # 정규식: 숫자 + 마침표/괄호 + 공백(선택)
            match = re.match(r'^(\d+)[\.\)]', text)
            
            if match:
                if current_prob:
                    problems.append(current_prob)
                
                prob_no = match.group(1)
                current_prob = {
                    "no": prob_no,
                    "question": tag.get_text(separator="\n", strip=True),
                    "answer": "",
                    "images": []
                }
                # 문제 번호가 포함된 태그 내 이미지 체크
                img = tag.find('img')
                if img:
                    self._process_image(img, year, exam_round, prob_no, current_prob)
                continue

            if current_prob:
                # 정답 구역(moreless) 처리
                if tag.name == 'div' and ('moreLess' in tag.get('class', []) or 'moreless' in str(tag.get('class'))):
                    ans_content = tag.select_one('.moreless-content')
                    if ans_content:
                        current_prob["answer"] = ans_content.get_text(strip=True)
                        problems.append(current_prob)
                        current_prob = None
                    continue

                # 이미지 처리 전용 메서드 호출
                img = tag.find('img')
                if img:
                    self._process_image(img, year, exam_round, current_prob['no'], current_prob)

                # 내용 누적 (표 또는 텍스트)
                if tag.name == 'table':
                    rows = [" | ".join([td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]) for tr in tag.find_all('tr')]
                    current_prob["question"] += "\n[표]\n" + "\n".join(rows)
                elif text and tag.name != 'figure':
                    # 너무 짧은 의미 없는 텍스트 방지 및 중복 방지
                    if text not in current_prob["question"]:
                        current_prob["question"] += "\n" + text

        # 루프 종료 후 남은 마지막 문제 처리
        if current_prob and current_prob not in problems:
            problems.append(current_prob)

        return {
            "year": year,
            "round": exam_round,
            "url": url,
            "problems": problems
        }

    def _process_image(self, img_tag, year, round_val, no, prob_dict):
        """이미지 추출 및 다운로드 공통 로직"""
        img_src = img_tag.get('src') or img_tag.get('data-src') or img_tag.get('data-url')
        if img_src and 'img.png' in img_src: # 티스토리 이미지 엔진 주소 확인
            img_name = f"{year}_{round_val}_{no}_{int(time.time())}.png"
            path = self.download_image(img_src, img_name)
            if path: prob_dict["images"].append(path)

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