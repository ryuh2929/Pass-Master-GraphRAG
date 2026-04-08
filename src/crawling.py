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
        except: pass
        return ""

    def parse_post(self, url: str) -> Dict:
        res = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(res.text, 'lxml')
        
        # 제목 및 메타정보 추출
        title_tag = soup.select_one('.post-cover h1') or soup.find('h1')
        title_text = title_tag.get_text(strip=True) if title_tag else ""
        year = re.search(r'(\d{4})년', title_text)
        round_val = re.search(r'(\d)회', title_text)
        year, exam_round = (year.group(1) if year else "Unknown"), (round_val.group(1) if round_val else "Unknown")

        content = soup.select_one('.entry-content')
        if not content: return {}

        problems = []
        current_prob = None
        
        # [핵심] 순차 탐색을 위해 모든 직계 및 주요 컨테이너 자식들을 평면화하여 탐색
        elements = content.find_all(['p', 'table', 'div', 'figure'], recursive=True)
        
        visited_tags = set() # 중복 처리 방지

        for tag in elements:
            if tag in visited_tags or tag.find_parent('div', class_='moreless-content'):
                continue
            
            # 1. 문제 시작 감지 패턴 강화
            # 단순히 '숫자.'으로 시작하는 게 아니라, <b> 태그 안에 있거나 줄의 맨 앞에 있을 때만 인정
            text = tag.get_text(strip=True)
            match = re.match(r'^(\d+)[\.\)]', text)
            
            # "진짜" 문제 번호인지 확인 (정답 리스트인 "1. TTL" 등과 구분)
            is_new_problem = False
            if match:
                prob_no = int(match.group(1))
                # 이전 문제 번호보다 크거나, 1번인 경우만 신규 문제로 인정 (간단한 검증)
                if not current_prob or int(current_prob['no']) < prob_no or prob_no == 1:
                    # 표 안에 있는 숫자는 문제 번호로 취급하지 않음
                    if not tag.find_parent('table'):
                        is_new_problem = True

            if is_new_problem:
                if current_prob: problems.append(current_prob)
                current_prob = {
                    "no": str(prob_no),
                    "question": text,
                    "answer": "",
                    "images": []
                }
                visited_tags.add(tag)
                # 이미지 체크
                self._extract_images(tag, year, exam_round, current_prob)
                continue

            if current_prob:
                # 2. 정답 구역(moreless) 처리
                if tag.name == 'div' and ('moreLess' in tag.get('class', []) or 'btn-toggle-moreless' in str(tag)):
                    ans_div = tag.select_one('.moreless-content')
                    if ans_div:
                        current_prob["answer"] = ans_div.get_text(separator=" ", strip=True).replace("더보기", "")
                        problems.append(current_prob)
                        current_prob = None # 문제 종료
                        # moreless 내부 태그들 방문 처리
                        for child in tag.find_all(): visited_tags.add(child)
                    continue

                # 3. 본문 누적 (문제 지문)
                if tag.name == 'table':
                    rows = [" | ".join([td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]) for tr in tag.find_all('tr')]
                    current_prob["question"] += "\n[표]\n" + "\n".join(rows)
                elif tag.name == 'figure' or tag.find('img'):
                    self._extract_images(tag, year, exam_round, current_prob)
                else:
                    if text and text not in current_prob["question"]:
                        # "더보기" 글자 제거
                        clean_text = text.replace("더보기", "").strip()
                        if clean_text: current_prob["question"] += "\n" + clean_text
                
                visited_tags.add(tag)

        return {"year": year, "round": exam_round, "url": url, "problems": problems}

    def _extract_images(self, tag, year, round_val, prob_dict):
        imgs = tag.find_all('img')
        for img in imgs:
            src = img.get('src') or img.get('data-src')
            if src and src.startswith('http') and src not in [i.split('/')[-1] for i in prob_dict["images"]]:
                # 이미지 중복 체크: 이미 같은 URL이나 파일명이 있으면 스킵
                fname = f"{year}_{round_val}_{prob_dict['no']}_{hash(src)%10000}.png"
                path = self.download_image(src, fname)
                if path and path not in prob_dict["images"]:
                    prob_dict["images"].append(path)

    def run(self, urls: List[str]):
        for url in urls:
            print(f"🔎 파싱 중: {url}")
            data = self.parse_post(url)
            if data and data['problems']:
                # 합격률 표 같은 쓰레기 데이터 제거 (문제가 너무 적거나 형식이 이상한 경우)
                data['problems'] = [p for p in data['problems'] if len(p['question']) > 5]
                
                filename = f"exam_{data['year']}_{data['round']}.json"
                with open(os.path.join(self.output_dir, filename), 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"✅ 저장 완료: {filename} ({len(data['problems'])}문제)")
            time.sleep(1)

if __name__ == "__main__":
    urls = ["https://chobopark.tistory.com/554", "https://chobopark.tistory.com/540"]
    crawler = ExamCrawler()
    crawler.run(urls)