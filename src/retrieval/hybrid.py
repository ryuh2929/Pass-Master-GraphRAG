from dataclasses import dataclass
import re
from typing import Pattern

from rank_bm25 import BM25Okapi

from src.retrieval.tokenizer import build_bm25_document_text, tokenize_for_bm25


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


CONCEPT_HINT_PATTERNS: dict[str, list[Pattern[str]]] = {
    "sql": [re.compile(r"sql|8장 sql|select|insert|delete|join|테이블|질의", re.IGNORECASE)],
    "c": [re.compile(r"c언어|c\s*언어|포인터|구조체", re.IGNORECASE)],
    "java": [re.compile(r"java|자바", re.IGNORECASE)],
    "python": [re.compile(r"python|파이썬", re.IGNORECASE)],
}


@dataclass
class ConceptDocument:
    section_id: str
    title: str
    chapter: str
    document: str
    practical_dates: list[str]
    hint_types: set[str]
    tokens: list[str]


@dataclass
class HybridCandidate:
    concept: ConceptDocument
    vector_score: float
    bm25_score: float
    final_score: float


def detect_code_hints(text: str) -> set[str]:
    hints = set()
    for hint, patterns in CODE_HINT_PATTERNS.items():
        if any(pattern.search(text) for pattern in patterns):
            hints.add(hint)
    return hints


class BM25ConceptRanker:
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
    question_hints = detect_code_hints(question_text)
    if question_hints:
        vector_weight = code_vector_weight
        bm25_weight = code_bm25_weight

    candidate_ids = set(vector_scores) | set(bm25_scores)
    valid_vector_scores = {}
    valid_bm25_scores = {}

    for concept_id in candidate_ids:
        concept = ranker.get_concept(concept_id)
        if not concept:
            continue

        if question_hints and question_hints.isdisjoint(concept.hint_types):
            continue

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
