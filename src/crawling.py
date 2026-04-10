import requests
from bs4 import BeautifulSoup
import json
import os
import re
import time
import random
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

        # 1. 실기 날짜 매핑 데이터
        practical_date_map = {
            "2020-1": "20.5", "2020-2": "20.7", "2020-3": "20.10", "2020-4": "20.11",
            "2021-1": "21.4", "2021-2": "21.7", "2021-3": "21.10",
            "2022-1": "22.5", "2022-2": "22.7", "2022-3": "22.10",
            "2023-1": "23.4", "2023-2": "23.7", "2023-3": "23.10",
            "2024-1": "24.4", "2024-2": "24.7", "2024-3": "24.10",
            "2025-1": "25.4", "2025-2": "25.7", "2025-3": "25.11"
        }
        
        # 제목 및 메타정보 추출
        title_tag = soup.select_one('.post-cover h1') or soup.find('h1')
        title_text = title_tag.get_text(strip=True) if title_tag else ""
        year_match = re.search(r'(\d{4})년', title_text)
        round_match = re.search(r'(\d)회', title_text)
        year = year_match.group(1) if year_match else "Unknown"
        exam_round = round_match.group(1) if round_match else "Unknown"
        practical_dates = practical_date_map.get(f"{year}-{exam_round}", "Unknown")
        content = soup.select_one('.entry-content')
        if not content: return {}

        problems = []
        current_prob = None
        last_prob_no = 0  # 마지막으로 확정된 문제 번호 추적
        
        elements = content.find_all(['p', 'table', 'div', 'figure', 'pre'], recursive=True)
        visited_tags = set()
        downloaded_urls = set()
        is_parsing_finished = False

        more_blocks = content.find_all('div', attrs={'data-ke-type': 'moreLess'})

        for tag in elements:
            if is_parsing_finished:
                break

            # 이미 처리된 태그나 정답 박스 안의 내용은 지문으로 읽지 않음
            if tag in visited_tags:
                continue
            
            text = tag.get_text(strip=True)
            if not text and not tag.find('img'): continue

            # 신규 문제 판단 로직: last_prob_no를 활용한 엄격한 검증
            is_new_problem = False
            match = re.search(r'^(\d+)[\.\)]', text)
            
            if match:
                # 번호가 정확히 다음 번호이거나, 첫 시작(1번)인 경우만 인정
                prob_no_val = int(match.group(1))
                
                # 1. 아예 처음 시작하는 경우 (1번)
                if last_prob_no == 0 and prob_no_val == 1:
                    is_new_problem = True
                
                # 2. 현재 수집 중인 문제가 있고, '정답(answer)'이 이미 채워진 상태에서 다음 번호가 온 경우
                # 정답 내부의 '1. ON' 등은 last_prob_no보다 작으므로 여기서 걸러짐
                elif current_prob and current_prob['answer'] and prob_no_val == last_prob_no + 1:
                    is_new_problem = True
                
                # 3. [예외] 정답은 아직 못 찾았지만, 태그가 <b>나 <strong>으로 감싸진 '제목형' 번호인 경우
                elif current_prob and not current_prob['answer'] and prob_no_val == last_prob_no + 1:
                    is_bold = tag.name in ['b', 'strong'] or tag.find(['b', 'strong'])
                    if is_bold:
                        is_new_problem = True

            if is_new_problem:
                # 새로운 문제를 만들기 전에 이전 문제 저장 (번호가 넘어갔으므로)
                if current_prob:
                    problems.append(current_prob)
                
                last_prob_no = int(match.group(1)) # 현재 번호 확정
                current_prob = {
                    "no": str(last_prob_no),
                    "question": text,
                    "answer": "",
                    "images": []
                }
                visited_tags.add(tag)
                self._extract_images(tag, year, exam_round, current_prob, downloaded_urls)
                continue

            if current_prob:

                if more_blocks:
                    if tag.get('data-ke-type') == 'moreLess' or 'moreLess' in tag.get('class', []) or tag.select_one('.btn-toggle-moreless'):
                        # 내부에서 실제 정답 텍스트가 있는 곳을 찾음
                        content_div = tag.find(class_='moreless-content')
                        if content_div:
                            current_prob["answer"] = content_div.get_text(strip=True)
                            
                            # 중요: 이 구역 안의 모든 자식 태그들을 방문 처리해서 지문에 안 섞이게 함
                            # for child in tag.find_all(recursive=True):
                            #     visited_tags.add(child)
                            # Note: 중첩된 HTML 구조(코드 블록 내 정답 포함 등)에서 
                            # 데이터 누락을 방지하기 위해 중복 방문을 허용함.
                            
                            if current_prob['no'] == "20":
                                is_parsing_finished = True
                                problems.append(current_prob)
                                current_prob = None
                        continue # 정답 처리 끝났으니 다음 태그로 이동

                # 2. 소스코드(colorscripter) 특수 처리
                if 'colorscripter-code' in str(tag.get('class', [])) or tag.select_one('table.colorscripter-code-table'):
                    tds = tag.find_all('td')
                    if len(tds) >= 2:
                        code_content = tds[1].get_text(separator="\n", strip=True)
                        current_prob["question"] += f"\n\n[Source Code]\n{code_content}"
                    
                    visited_tags.add(tag)
                    for child in tag.find_all():
                        visited_tags.add(child)
                    continue

                # 3. 이미지 및 일반 텍스트 누적
                if tag.name == 'figure' or tag.find('img'):
                    self._extract_images(tag, year, exam_round, current_prob, downloaded_urls)
                    visited_tags.add(tag)
                    continue
                
                if tag.name == 'table':
                    rows = [" | ".join([td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]) for tr in tag.find_all('tr')]
                    current_prob["question"] += "\n[표]\n" + "\n".join(rows)
                    visited_tags.add(tag)
                    for child in tag.find_all(): visited_tags.add(child)
                else:
                    clean_text = text.replace("더보기", "").strip()
                    if clean_text and clean_text not in current_prob["question"]:
                        # 이미 정답이 채워졌는데 또 텍스트가 들어오려고 하면 차단 (다음 문제를 기다림)
                        if not current_prob["answer"]:
                            current_prob["question"] += "\n" + clean_text
                
                visited_tags.add(tag)

        # 마지막 문제가 20번이 아닐 경우를 대비해 루프 종료 후 남은 문제 추가
        if current_prob and current_prob not in problems:
            problems.append(current_prob)

        return {"year": year, "round": exam_round, "practical_dates": practical_dates, "url": url, "problems": problems}

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
            time.sleep(random.uniform(0.3, 0.5))

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