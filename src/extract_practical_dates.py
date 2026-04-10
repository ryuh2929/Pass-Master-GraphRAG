import json

# JSON 파일 로드
with open('data/processed/processed_chunks.json', 'r', encoding='utf-8') as f:
    data_list = json.load(f)

# "practical_dates" 목록만 평면화(flatten)해서 추출
all_practical_dates = [
    date 
    for item in data_list 
    if isinstance(item, dict)  # item이 딕셔너리인지 확인
    for date in item.get("metadata", {}).get("practical_dates", []) # 키가 없으면 빈 리스트 반환
]

# 중복 제거 set 활용
unique_dates = sorted(list(set(all_practical_dates)))

print(unique_dates)