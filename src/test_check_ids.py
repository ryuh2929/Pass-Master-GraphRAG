import json
import os

def check_missing_ids(json_path):
    if not os.path.exists(json_path):
        print(f"[!] 파일을 찾을 수 없습니다: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # 1. 추출된 ID 목록 (정수형으로 변환)
    found_ids = sorted([int(c["metadata"]["id"]) for c in chunks])
    
    # 2. 전체 있어야 할 ID 목록 (1~301)
    all_expected_ids = set(range(1, 302))
    found_ids_set = set(found_ids)

    # 3. 차집합으로 누락된 번호 찾기
    missing = sorted(list(all_expected_ids - found_ids_set))
    
    # 4. 중복된 번호가 있는지 확인 (혹시 모르니)
    duplicates = set([x for x in found_ids if found_ids.count(x) > 1])

    print("="*30)
    print(f"[*] 총 탐지된 섹션: {len(found_ids)}개")
    if missing:
        print(f"[!] 누락된 번호 ({len(missing)}개): {missing}")
    else:
        print("[V] 누락된 번호가 없습니다!")
        
    if duplicates:
        print(f"[!] 중복된 번호: {list(duplicates)}")
    print("="*30)

if __name__ == "__main__":
    check_missing_ids("data/processed/processed_chunks.json")