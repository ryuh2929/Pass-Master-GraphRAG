import pdfplumber
import glob
import re
import json
import os

def chunk_pdf_text(full_text, valid_dates):
    """
    full_text: pdfplumber로 추출한 전체 텍스트
    valid_dates: 추출한 34개의 기출 날짜 세트 (set)
    """

    full_text = full_text.replace('\x07', ' ')
    # 1. 소제목(001) 위치만 먼저 찾습니다. (이건 매우 빠릅니다)
    section_pattern = re.compile(r'(?:^|\n)(\d{3})\s+([^\n]+?)\s+([A-C])(?:\s|\n|$)')
    matches = list(section_pattern.finditer(full_text))
    chunks = []

    for i in range(len(matches)):
        # 2. 이번 섹션의 시작점 결정
        # 기본값은 소제목 시작점이지만, 그 윗부분(약 100자 정도)에 날짜가 있는지 확인합니다.
        current_match_start = matches[i].start()
        lookback_window = full_text[max(0, current_match_start-100):current_match_start]
        
        # 윗부분에서 가장 먼저 나오는 날짜의 위치를 찾음
        date_matches = list(re.finditer(r'\d{2}\.(?:1[0-2]|[1-9])', lookback_window))
        if date_matches:
            # 날짜가 있다면 그 날짜의 시작점으로 start_idx를 당깁니다.
            start_idx = max(0, current_match_start - 100) + date_matches[0].start()
        else:
            start_idx = current_match_start

        # 3. 끝점 결정 (다음 섹션의 시작점 혹은 전체 끝)
        if i + 1 < len(matches):
            next_match_start = matches[i+1].start()
            # 다음 섹션 윗부분에 날짜가 있는지 또 확인해서 그 전까지만 끊음
            next_lookback = full_text[max(0, next_match_start-100):next_match_start]
            next_date_matches = list(re.finditer(r'\d{2}\.(?:1[0-2]|[1-9])', next_lookback))
            if next_date_matches:
                end_idx = max(0, next_match_start - 100) + next_date_matches[0].start()
            else:
                end_idx = next_match_start
        else:
            end_idx = len(full_text)
        
        section_text = full_text[start_idx:end_idx].strip()
        
        # 메타데이터 추출 (matches[i] 그룹 사용)
        item_id = matches[i].group(1)
        title = matches[i].group(2).strip()
        importance = matches[i].group(3).strip()
        
        # 본문 내 날짜 추출 (이제 본문 안에 본인의 날짜가 포함되어 있음)
        found_in_section = set(re.findall(r'(?<!\d)\d{2}\.(?:1[0-2]|[1-9])(?!\d)', section_text))
        actual_exam_dates = sorted(list(found_in_section.intersection(valid_dates)))
        
        # 결과 딕셔너리 생성
        chunks.append({
            "document": section_text,  # LLM이 읽을 본문
            "metadata": {
                "id": item_id,
                "chapter": get_chapter_name(item_id),
                "title": title,
                "importance": importance,
                "exam_dates": actual_exam_dates,
                "occurrence_count": len(actual_exam_dates),
                "is_practical": any(d in ['20.5','20.7','20.10','20.11','21.4','21.7','21.10','22.5','22.10','23.4','23.10','24.4','24.7', '24.10','25.4','25.7','25.11'] for d in actual_exam_dates) # 실기 날짜 포함 여부
            }
        })

        # [298-299번 섹션 분리 및 정제 패치]
    final_chunks = []
    for chunk in chunks:
        # 298번에 299번 내용이 침범한 경우 (문자열 탐색)
        if chunk["metadata"]["id"] == "298" and "299 C" in chunk["document"]:
            # 1. 298번 본문에서 299번 시작 지점 전까지만 자르기
            split_target = "340503 필기 25.8" # 299번 날짜 시작점 혹은 구분자
            parts = chunk["document"].split(split_target)
            chunk["document"] = parts[0].strip()
            final_chunks.append(chunk)
            
            # 2. 누락된 299번 수동 생성
            new_299 = {
                "document": f"{split_target}{parts[1]}".strip() if len(parts) > 1 else "299번 본문 유실 확인 필요",
                "metadata": {
                    "id": "299",
                    "chapter": "12장 제품 소프트웨어 패키징",
                    "title": "소프트웨어 버전 관리 도구 - 분산 저장소 방식",
                    "importance": "C",
                    "exam_dates": ["21.5", "25.8"],
                    "occurrence_count": 2,
                    "is_practical": False
                }
            }
            final_chunks.append(new_299)
        else:
            final_chunks.append(chunk)

    return final_chunks

def get_chapter_name(item_id_str):
    try:
        item_id = int(item_id_str)
        if 1 <= item_id <= 33: return "1장 요구사항 확인"
        elif 34 <= item_id <= 84: return "2장 데이터 입출력 구현"
        elif 85 <= item_id <= 88: return "3장 통합 구현"
        elif 89 <= item_id <= 116: return "4장 서버 프로그램 구현"
        elif 117 <= item_id <= 126: return "5장 인터페이스 구현"
        elif 127 <= item_id <= 130: return "6장 화면 설계"
        elif 131 <= item_id <= 151: return "7장 애플리케이션 테스트 관리"
        elif 152 <= item_id <= 174: return "8장 SQL 응용"
        elif 175 <= item_id <= 210: return "9장 소프트웨어 개발 보안 구축"
        elif 211 <= item_id <= 234: return "10장 프로그래밍 언어 활용"
        elif 235 <= item_id <= 291: return "11장 응용 SW 기초 기술 활용"
        elif 292 <= item_id <= 301: return "12장 제품 소프트웨어 패키징"
        else: return "기타"
    except ValueError:
        return "알 수 없음"

def extract_full_text(file_pattern):
    """
    PDF 파일을 열어서 모든 페이지의 텍스트를 하나의 문자열로 합칩니다.
    """
    full_text = ""

    # 와일드카드 패턴에 맞는 모든 파일 경로 리스트 가져오기
    pdf_files = glob.glob(file_pattern)

    # 1단 구성 페이지 리스트 (0부터 시작하는 인덱스)
    single_col_pages = {8, 9, 10, 11, 21, 22, 23, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98}
    
    if not pdf_files:
        print(f"[!] 파일을 찾을 수 없습니다: {file_pattern}")
        return ""

    print(f"[*] 총 {len(pdf_files)}개의 파일을 찾았습니다: {pdf_files}")
    
    for pdf_path in pdf_files:
        # pdfplumber.open()으로 파일을 먼저 열어서 페이지 수를 확인한 후, 각 페이지마다 1단 또는 2단으로 텍스트를 추출
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"[*] '{pdf_path}' 분석 중 (총 {total_pages}페이지)...")

            for i, page in enumerate(pdf.pages):
                # 1단 구성 페이지인 경우
                if i in single_col_pages:
                    page_text = page.extract_text() or ""
                    full_text += f"\n{page_text}\n"
                # 2단 구성 페이지인 경우
                else:
                    # 페이지의 너비와 높이 구하기
                    width, height = page.width, page.height
                    # 좌측과 우측을 각각 크롭해서 텍스트 추출
                    left = page.within_bbox((0, 0, width / 2, height)).extract_text() or ""
                    right = page.within_bbox((width / 2, 0, width, height)).extract_text() or ""
                    full_text += f"\n{left}\n{right}\n"
                
                # 진행률 표시 (20페이지마다 또는 마지막 페이지일 때)
                if (i + 1) % 20 == 0 or (i + 1) == total_pages:
                    print(f"[*] 진행도: {i + 1}/{total_pages} 페이지 완료")

    # 특수문자 제거 및 정제
    full_text = full_text.replace('\x07', ' ')
    print(f"[*] 추출 완료! 총 글자 수: {len(full_text)}자")
    return full_text

if __name__ == "__main__":
    # 1. PDF 파일 경로 설정 (data 폴더 안에 넣어두세요)
    PATH_PATTERN = "data/*.pdf"
    OUTPUT_DIR = "data"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "processed_chunks.json")

    # 2. 폴더가 없으면 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 3. 텍스트 추출 및 청킹 실행
    print("[*] PDF 텍스트 추출 및 청킹 프로세스 시작...")
    result_text = extract_full_text(PATH_PATTERN)

    print("\n--- 추출된 전체 텍스트 일부 ---")
    print(result_text[:100])  # 처음 100자만 출력 (디버깅용)
    
    # 유효 출제 날짜 세트
    valid_dates = set(['20.5', '20.6', '20.7', '20.8', '20.9', '20.10', '20.11', '21.3', '21.4', '21.5', '21.7', '21.8', '21.10', '22.3', '22.4', '22.5', '22.7', '22.10', '23.2', '23.4', '23.5', '23.7', '23.10', '24.2', '24.4', '24.5', '24.7', '24.10', '25.2', '25.4', '25.5', '25.7', '25.8', '25.11'])

    chunks = chunk_pdf_text(result_text, valid_dates)
    
    # 4. JSON 저장 (ensure_ascii=False 필수: 한글 깨짐 방지)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # 5. 최종 데이터 분석 결과 출력
    print("\n" + "="*30)
    print(f"[*] 처리 완료! 총 {len(chunks)}개의 섹션이 저장되었습니다.")
    print(f"[*] 저장 위치: {OUTPUT_FILE}")
    print("="*30)

    print("\n--- 첫 번째 청크 예시 ---")
    if chunks:   # 청크가 존재할 때만 출력
        print(chunks[0])
        print(f"총 {len(chunks)}개")
    else:
        print("[!] 청크가 생성되지 않았습니다.")