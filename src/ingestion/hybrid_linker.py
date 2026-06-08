"""Question-Concept 자동 연결용 hybrid linker.

이 모듈은 앱 실행 중 사용자 검색에 쓰는 retriever가 아니라, ingestion 단계에서
Neo4j의 VERIFIED_MENTIONS 관계를 새로 만들 때 사용합니다.
평가셋 JSON은 정답지이므로 여기서 참조하지 않고, Question 원문/정답과 Concept
본문만으로 vector 점수와 BM25 점수를 합산합니다.
"""

from dataclasses import dataclass
import re
from typing import Pattern

from rank_bm25 import BM25Okapi

from src.retrieval.tokenizer import build_bm25_document_text, tokenize_for_bm25


# 문제 원문에서 SQL/C/Java/Python 힌트를 감지하는 규칙입니다.
# 코드/SQL 문제는 표면 키워드가 중요하므로, 힌트가 잡히면 BM25 가중치를 높이고
# 다른 언어 Concept로 새는 후보를 줄입니다.
CODE_HINT_PATTERNS: dict[str, list[Pattern[str]]] = {
    "sql": [
        re.compile(pattern, re.IGNORECASE)
        for pattern in [
            r"\bsql\b",
            r"\bselect\b",
            r"\binsert\b",
            r"\bdelete\b",
            r"\bupdate\b",
            r"\bcreate\b",
            r"\bjoin\b",
            r"\bgroup\s+by\b",
            r"\border\s+by\b",
        ]
    ],
    "c": [
        re.compile(pattern, re.IGNORECASE)
        for pattern in [
            r"c\s*(언어|코드|프로그램)",
            r"#\s*include",
            r"\bprintf\s*\(",
            r"\bscanf\s*\(",
            r"\bmalloc\s*\(",
            r"\bfree\s*\(",
        ]
    ],
    "java": [
        re.compile(pattern, re.IGNORECASE)
        for pattern in [
            r"\bjava\b",
            r"자바",
            r"\bpublic\s+class\b",
            r"\bstring\s*\[\s*\]\s*args\b",
            r"\bextends\b",
            r"\bimplements\b",
        ]
    ],
    "python": [
        re.compile(pattern, re.IGNORECASE)
        for pattern in [
            r"\bpython\b",
            r"파이썬",
            r"\bdef\s+[a-z_][a-z0-9_]*\s*\(",
            r"\brange\s*\(",
        ]
    ],
}


# Concept가 어떤 코드/SQL 계열인지 판별하는 규칙입니다.
# Question에서 언어 힌트가 감지되면 이 값이 맞는 Concept만 후보로 남깁니다.
CONCEPT_HINT_PATTERNS: dict[str, list[Pattern[str]]] = {
    "sql": [re.compile(r"sql|8장 sql|select|insert|delete|join|테이블|질의", re.IGNORECASE)],
    "c": [re.compile(r"c언어|c\s*언어|포인터|구조체", re.IGNORECASE)],
    "java": [re.compile(r"java|자바", re.IGNORECASE)],
    "python": [re.compile(r"python|파이썬", re.IGNORECASE)],
}


@dataclass
class ConceptDocument:
    """BM25 인덱싱과 날짜 검증에 필요한 Concept 표현입니다."""

    section_id: str
    title: str
    chapter: str
    document: str
    practical_dates: list[str]
    hint_types: set[str]
    tokens: list[str]


@dataclass
class HybridCandidate:
    """최종 연결 후보입니다. 저장되는 관계에는 세 점수를 모두 남깁니다."""

    concept: ConceptDocument
    vector_score: float
    bm25_score: float
    final_score: float


def detect_code_hints(text: str) -> set[str]:
    """Question 원문에서 코드/SQL 계열 힌트를 추출합니다."""

    hints = set()
    for hint, patterns in CODE_HINT_PATTERNS.items():
        if any(pattern.search(text) for pattern in patterns):
            hints.add(hint)
    return hints


class BM25ConceptRanker:
    """Concept 목록을 BM25 문서로 인덱싱하고 Question별 BM25 점수를 계산합니다."""

    def __init__(self, concepts: list[dict]):
        self.concepts = [self._build_concept_document(concept) for concept in concepts]
        self.bm25 = BM25Okapi([concept.tokens for concept in self.concepts])

    def get_scores(self, question_text: str) -> dict[str, float]:
        query_tokens = tokenize_for_bm25(question_text)
        raw_scores = self.bm25.get_scores(query_tokens)

        return {
            concept.section_id: float(score)
            for concept, score in zip(self.concepts, raw_scores)
            if score > 0
        }

    def get_concept(self, section_id: str) -> ConceptDocument | None:
        for concept in self.concepts:
            if concept.section_id == section_id:
                return concept
        return None

    def _build_concept_document(self, concept: dict) -> ConceptDocument:
        title = str(concept.get("title") or "")
        chapter = str(concept.get("chapter") or "")
        document = str(concept.get("document") or "")
        # 제목은 Concept 식별력이 높으므로 tokenizer 이전 단계에서 반복 가중치를 줍니다.
        text = build_bm25_document_text(title=title, chapter=chapter, content=document)

        return ConceptDocument(
            section_id=str(concept["section_id"]),
            title=title,
            chapter=chapter,
            document=document,
            practical_dates=[str(value).strip() for value in concept.get("practical_dates") or []],
            hint_types=_detect_concept_hints(f"{title} {chapter} {document}"),
            tokens=tokenize_for_bm25(text),
        )


def build_hybrid_candidates(
    *,
    question_text: str,
    practical_date: str,
    vector_scores: dict[str, float],
    bm25_scores: dict[str, float],
    ranker: BM25ConceptRanker,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    code_vector_weight: float = 0.3,
    code_bm25_weight: float = 0.7,
) -> list[HybridCandidate]:
    """Vector 후보와 BM25 후보를 합쳐 최종 연결 후보를 만듭니다.

    처리 순서:
    1. 후보 union
    2. 코드/SQL 힌트가 있으면 맞는 계열 Concept만 유지
    3. practical_date가 맞지 않는 후보 제거
    4. vector/BM25 점수를 각각 0~1로 정규화
    5. 일반 문제와 코드 문제의 가중치를 다르게 적용
    """

    question_hints = detect_code_hints(question_text)
    if question_hints:
        # 코드/SQL 문제는 의미 유사도보다 키워드 매칭이 더 믿을 만한 경우가 많습니다.
        vector_weight = code_vector_weight
        bm25_weight = code_bm25_weight

    candidate_ids = set(vector_scores) | set(bm25_scores)
    valid_vector_scores = {}
    valid_bm25_scores = {}

    for concept_id in candidate_ids:
        concept = ranker.get_concept(concept_id)
        if not concept:
            continue

        # 언어 힌트가 있는 문제는 다른 언어 Concept로 연결되지 않도록 강하게 제한합니다.
        if question_hints and question_hints.isdisjoint(concept.hint_types):
            continue

        # 현재 평가 기준은 "출제 날짜 정보가 맞다"는 가정이므로 날짜 불일치 후보는 제외합니다.
        if not _date_matches(practical_date, concept.practical_dates):
            continue

        valid_vector_scores[concept_id] = vector_scores.get(concept_id, 0.0)
        valid_bm25_scores[concept_id] = bm25_scores.get(concept_id, 0.0)

    normalized_vector = _normalize_scores(valid_vector_scores)
    normalized_bm25 = _normalize_scores(valid_bm25_scores)

    candidates = []
    for concept_id in valid_vector_scores:
        concept = ranker.get_concept(concept_id)
        if not concept:
            continue

        final_score = (
            vector_weight * normalized_vector.get(concept_id, 0.0)
            + bm25_weight * normalized_bm25.get(concept_id, 0.0)
        )
        candidates.append(
            HybridCandidate(
                concept=concept,
                vector_score=vector_scores.get(concept_id, 0.0),
                bm25_score=bm25_scores.get(concept_id, 0.0),
                final_score=final_score,
            )
        )

    return sorted(candidates, key=lambda candidate: candidate.final_score, reverse=True)


def _detect_concept_hints(text: str) -> set[str]:
    hints = set()
    for hint, patterns in CONCEPT_HINT_PATTERNS.items():
        if any(pattern.search(text) for pattern in patterns):
            hints.add(hint)
    return hints


def _date_matches(practical_date: str, concept_dates: list[str]) -> bool:
    target = str(practical_date or "").strip()
    return bool(target) and any(target == str(date).strip() for date in concept_dates)


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """스케일이 다른 vector 점수와 BM25 점수를 합산하기 위해 0~1 범위로 맞춥니다."""

    if not scores:
        return {}

    min_score = min(scores.values())
    max_score = max(scores.values())
    if max_score == min_score:
        return {key: 1.0 if value > 0 else 0.0 for key, value in scores.items()}

    return {
        key: (value - min_score) / (max_score - min_score)
        for key, value in scores.items()
    }
