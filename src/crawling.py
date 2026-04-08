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
        path = os.path.join(self.image_dir, filename)
        
        # 파일명이 고정되었으므로 이제 이 체크가 정상 작동합니다.
        if os.path.exists(path):
            return path

        try:
            res = requests.get(img_url, headers=self.headers, timeout=10)
            if res.status_code == 200:
                with open(path, 'wb') as f:
                    f.write(res.content)
                return path
        except Exception as e:
            print(f" ❌ 이미지 다운로드 실패: {e}")
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
        visited_tags = set()
        downloaded_urls = set()
        is_parsing_finished = False

        for tag in elements:
            # 파싱이 종료되었다면 더 이상 읽지 않음
            if is_parsing_finished:
                break

            if tag in visited_tags or tag.find_parent('div', class_='moreless-content'):
                continue
            
            text = tag.get_text(strip=True)
            if not text and not tag.find('img'): continue

            # [핵심 수정] 신규 문제 판단 로직 강화
            is_new_problem = False
            match = re.match(r'^(\d+)[\.\)]', text)
            
            if match:
                prob_no = match.group(1)
                # 1. 현재 수집 중인 문제가 없을 때만 신규 문제로 인정
                # 2. 혹은 현재 수집 중인 문제가 있더라도, 번호 차이가 크거나 명확한 지문일 때 (선택적)
                if current_prob is None:
                    is_new_problem = True
                else:
                    # [예외] 만약 '더보기'를 못 만나서 닫히지 않은 상태에서 진짜 다음 번호(현재+1)가 나왔다면?
                    if int(prob_no) == int(current_prob['no']) + 1:
                        problems.append(current_prob)
                        is_new_problem = True

            if is_new_problem:
                current_prob = {
                    "no": prob_no,
                    "question": text,
                    "answer": "",
                    "images": []
                }
                visited_tags.add(tag)
                self._extract_images(tag, year, exam_round, current_prob, downloaded_urls)
                continue

            if current_prob:
                # 1. 정답 구역(moreless) 처리 -> 여기서 문제를 확실히 닫음
                if 'moreLess' in tag.get('class', []) or tag.select_one('.btn-toggle-moreless'):
                    ans_div = tag.select_one('.moreless-content')
                    if ans_div:
                        current_prob["answer"] = ans_div.get_text(separator="\n", strip=True)
                        problems.append(current_prob)

                        # 20번 정답을 찾았다면 종료 준비
                        if current_prob['no'] == "20":
                            is_parsing_finished = True

                        current_prob = None # 문제를 닫음 (다음 숫자 패턴을 새 문제로 인식 가능하게 함)
                        downloaded_urls = set() # 다음 문제에서 이미지 중복 방지 위해 초기화
                    continue

                # 2. 소스코드(colorscripter) 특수 처리
                if 'colorscripter-code' in str(tag.get('class', [])):
                    # 줄번호가 섞이지 않게 코드 본문만 추출
                    if 'colorscripter-code' in str(tag.get('class', [])) or tag.select_one('table.colorscripter-code-table'):
                        # 테이블 구조에서 줄번호(첫 번째 td)를 제외하고 실제 코드(두 번째 td)만 가져옵니다.
                        tds = tag.find_all('td')
                        if len(tds) >= 2:
                            # 두 번째 td가 실제 소스코드 영역입니다.
                            code_content = tds[1].get_text(separator="\n", strip=True)
                            current_prob["question"] += f"\n\n[Source Code]\n{code_content}"
                        
                        # 해당 태그와 그 자식들을 방문 처리하여 중복 방지
                        visited_tags.add(tag)
                        for child in tag.find_all():
                            visited_tags.add(child)
                        continue

                # 3. 이미지 및 일반 텍스트 누적
                if tag.name == 'figure' or tag.find('img'):
                    self._extract_images(tag, year, exam_round, current_prob, downloaded_urls)
                
                if tag.name == 'table' and 'colorscripter' not in str(tag.get('class')):
                    rows = [" | ".join([td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]) for tr in tag.find_all('tr')]
                    current_prob["question"] += "\n[표]\n" + "\n".join(rows)
                else:
                    clean_text = text.replace("더보기", "").strip()
                    if clean_text and clean_text not in current_prob["question"]:
                        current_prob["question"] += "\n" + clean_text
                
                visited_tags.add(tag)

        return {"year": year, "round": exam_round, "url": url, "problems": problems}

    def _extract_images(self, tag, year, round_val, prob_dict, downloaded_urls):
        # find_all 대신 현재 태그가 img인지 확인 + 자식들 중 img 탐색
        imgs = tag.find_all('img') if tag.name != 'img' else [tag]
        
        for img in imgs:
            src = img.get('src') or img.get('data-src')
            
            # 필터링 및 중복 URL 체크
            if not src or not src.startswith('http') or 'tistory_admin' in src:
                continue
            
            # [핵심] 이미 이 문제에서 처리한 URL이면 스킵
            if src in downloaded_urls:
                continue

            # 파일명 결정 (이제 URL이 다를 때만 인덱스가 증가함)
            img_idx = len(prob_dict["images"]) + 1
            fname = f"{year}_{round_val}_{prob_dict['no']}_{img_idx}.png"
            
            path = self.download_image(src, fname)
            
            if path:
                if path not in prob_dict["images"]:
                    prob_dict["images"].append(path)
                downloaded_urls.add(src) # 처리 완료된 URL 등록

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
    urls = ["https://chobopark.tistory.com/196", 
            "https://chobopark.tistory.com/195", 
            "https://chobopark.tistory.com/194", 
            "https://chobopark.tistory.com/192", 
            "https://chobopark.tistory.com/191", 
            "https://chobopark.tistory.com/210", 
            "https://chobopark.tistory.com/217", 
            "https://chobopark.tistory.com/271", 
            "https://chobopark.tistory.com/423", 
            "https://chobopark.tistory.com/424", 
            "https://chobopark.tistory.com/372", 
            "https://chobopark.tistory.com/420", 
            "https://chobopark.tistory.com/453", 
            "https://chobopark.tistory.com/476", 
            "https://chobopark.tistory.com/483", 
            "https://chobopark.tistory.com/495", 
            "https://chobopark.tistory.com/540", 
            "https://chobopark.tistory.com/554", 
            "https://chobopark.tistory.com/558"
            ]
    crawler = ExamCrawler()
    crawler.run(urls)