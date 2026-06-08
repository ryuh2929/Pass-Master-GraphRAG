import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

from langchain_neo4j import Neo4jGraph

from src.ingestion.hybrid_linker import (
    BM25ConceptRanker,
    HybridCandidate,
    LANGUAGE_CONCEPT_IDS,
    build_hybrid_candidates,
)


DEFAULT_GOLD_PATH = Path("data/evaluation/question_language_gold.seed.json")


@dataclass
class GoldCase:
    problem_id: str
    primary_concept: str
    expected_language: str
    review_status: str
    concept_source: str


@dataclass
class QuestionRecord:
    problem_id: str
    question: str
    answer: str
    practical_date: str
    embedding: list[float]


@dataclass
class StrategyResult:
    strategy: str
    problem_id: str
    expected_concept: str
    expected_language: str
    predicted_concept: str | None
    predicted_language_match: bool
    exact_match: bool
    missing_question: bool
    missing_candidate: bool
    final_score: float | None = None
    vector_score: float | None = None
    bm25_score: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DB 관계를 수정하지 않고 Question-Concept 연결 전략별 정확도를 비교합니다."
    )
    parser.add_argument(
        "--gold",
        default=str(DEFAULT_GOLD_PATH),
        help="평가셋 JSON 경로입니다. 기본값은 data/evaluation/question_language_gold.seed.json",
    )
    parser.add_argument(
        "--include-needs-review",
        action="store_true",
        help="review_status가 needs_review인 케이스도 평가에 포함합니다.",
    )
    parser.add_argument(
        "--vector-top-k",
        type=int,
        default=int(os.getenv("VECTOR_TOP_K", "500")),
        help="Neo4j vector index에서 가져올 Concept 후보 수입니다.",
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="선택한 전략의 오답 상세를 출력합니다.",
    )
    parser.add_argument(
        "--details-strategy",
        default="current_linker",
        choices=[
            "vector_only",
            "hybrid_bm25",
            "language_restricted_vector",
            "language_restricted_hybrid",
            "current_linker",
        ],
        help="--show-details로 상세 출력할 전략입니다.",
    )
    return parser.parse_args()


def create_graph() -> Neo4jGraph:
    load_dotenv()

    return Neo4jGraph(
        url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
    )


def load_gold_cases(gold_path: Path, include_needs_review: bool) -> list[GoldCase]:
    with gold_path.open("r", encoding="utf-8") as file:
        raw_cases = json.load(file)

    cases = []
    for item in raw_cases:
        review_status = item.get("review_status", "confirmed")
        if review_status != "confirmed" and not include_needs_review:
            continue

        cases.append(
            GoldCase(
                problem_id=str(item["problem_id"]),
                primary_concept=str(item["primary_concept"]),
                expected_language=str(item["expected_language"]),
                review_status=review_status,
                concept_source=str(item.get("concept_source", "")),
            )
        )

    return cases


def fetch_concepts(graph: Neo4jGraph) -> list[dict]:
    return graph.query(
        """
        MATCH (c:Concept)
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(ch:Chapter)
        RETURN
            c.section_id AS section_id,
            c.title AS title,
            c.document AS document,
            c.practical_dates AS practical_dates,
            ch.name AS chapter
        """
    )


def fetch_questions(graph: Neo4jGraph, problem_ids: list[str]) -> dict[str, QuestionRecord]:
    # 평가셋에 포함된 문제만 읽습니다. VERIFIED_MENTIONS 관계는 조회하지 않습니다.
    rows = graph.query(
        """
        UNWIND $problem_ids AS problem_id
        MATCH (q:Question {problem_id: problem_id})-[:HAS_QUESTION]-(e:Exam)
        WHERE q.embedding IS NOT NULL
        RETURN
            q.problem_id AS problem_id,
            q.question AS question,
            q.answer AS answer,
            q.embedding AS embedding,
            e.practical_dates AS practical_date
        """,
        {"problem_ids": problem_ids},
    )

    return {
        str(row["problem_id"]): QuestionRecord(
            problem_id=str(row["problem_id"]),
            question=str(row.get("question") or ""),
            answer=str(row.get("answer") or ""),
            practical_date=str(row.get("practical_date") or ""),
            embedding=row["embedding"],
        )
        for row in rows
    }


def fetch_vector_scores(graph: Neo4jGraph, embedding: list[float], top_k: int) -> dict[str, float]:
    rows = graph.query(
        """
        CALL db.index.vector.queryNodes('concept_index', $top_k, $vector)
        YIELD node AS c, score
        RETURN c.section_id AS section_id, score
        """,
        {"top_k": top_k, "vector": embedding},
    )

    return {
        str(row["section_id"]): float(row["score"])
        for row in rows
        if row["score"] >= 0
    }


def predict_all_strategies(
    *,
    graph: Neo4jGraph,
    cases: list[GoldCase],
    questions: dict[str, QuestionRecord],
    ranker: BM25ConceptRanker,
    vector_top_k: int,
) -> list[StrategyResult]:
    results = []

    for index, case in enumerate(cases, start=1):
        question = questions.get(case.problem_id)
        if not question:
            results.extend(_missing_question_results(case))
            continue

        vector_scores = fetch_vector_scores(graph, question.embedding, top_k=vector_top_k)
        bm25_text = f"{question.question} {question.answer}"
        bm25_scores = ranker.get_scores(bm25_text)

        strategies = {
            "vector_only": _build_unrestricted_candidates(
                practical_date=question.practical_date,
                vector_scores=vector_scores,
                bm25_scores={},
                ranker=ranker,
                vector_weight=1.0,
                bm25_weight=0.0,
            ),
            "hybrid_bm25": _build_unrestricted_candidates(
                practical_date=question.practical_date,
                vector_scores=vector_scores,
                bm25_scores=bm25_scores,
                ranker=ranker,
                vector_weight=0.6,
                bm25_weight=0.4,
            ),
            "language_restricted_vector": build_hybrid_candidates(
                question_text=question.question,
                practical_date=question.practical_date,
                vector_scores=vector_scores,
                bm25_scores=bm25_scores,
                ranker=ranker,
                vector_weight=1.0,
                bm25_weight=0.0,
                explicit_language_vector_weight=1.0,
                explicit_language_bm25_weight=0.0,
                code_vector_weight=1.0,
                code_bm25_weight=0.0,
            ),
            "language_restricted_hybrid": build_hybrid_candidates(
                question_text=question.question,
                practical_date=question.practical_date,
                vector_scores=vector_scores,
                bm25_scores=bm25_scores,
                ranker=ranker,
                vector_weight=0.6,
                bm25_weight=0.4,
                explicit_language_vector_weight=0.7,
                explicit_language_bm25_weight=0.3,
                code_vector_weight=0.3,
                code_bm25_weight=0.7,
            ),
            "current_linker": build_hybrid_candidates(
                question_text=question.question,
                practical_date=question.practical_date,
                vector_scores=vector_scores,
                bm25_scores=bm25_scores,
                ranker=ranker,
            ),
        }

        for strategy, candidates in strategies.items():
            results.append(_build_strategy_result(strategy, case, candidates))

        if index % 50 == 0:
            print(f"Evaluated dry-run cases: {index}/{len(cases)}")

    return results


def _build_unrestricted_candidates(
    *,
    practical_date: str,
    vector_scores: dict[str, float],
    bm25_scores: dict[str, float],
    ranker: BM25ConceptRanker,
    vector_weight: float,
    bm25_weight: float,
) -> list[HybridCandidate]:
    candidate_ids = set(vector_scores) | set(bm25_scores)
    valid_vector_scores = {}
    valid_bm25_scores = {}

    for concept_id in candidate_ids:
        concept = ranker.get_concept(concept_id)
        if not concept or not _date_matches(practical_date, concept.practical_dates):
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


def _build_strategy_result(
    strategy: str,
    case: GoldCase,
    candidates: list[HybridCandidate],
) -> StrategyResult:
    best = candidates[0] if candidates else None
    predicted_concept = best.concept.section_id if best else None

    return StrategyResult(
        strategy=strategy,
        problem_id=case.problem_id,
        expected_concept=case.primary_concept,
        expected_language=case.expected_language,
        predicted_concept=predicted_concept,
        predicted_language_match=_language_matches(case.expected_language, predicted_concept),
        exact_match=case.primary_concept == predicted_concept,
        missing_question=False,
        missing_candidate=best is None,
        final_score=best.final_score if best else None,
        vector_score=best.vector_score if best else None,
        bm25_score=best.bm25_score if best else None,
    )


def _missing_question_results(case: GoldCase) -> list[StrategyResult]:
    return [
        StrategyResult(
            strategy=strategy,
            problem_id=case.problem_id,
            expected_concept=case.primary_concept,
            expected_language=case.expected_language,
            predicted_concept=None,
            predicted_language_match=False,
            exact_match=False,
            missing_question=True,
            missing_candidate=False,
        )
        for strategy in [
            "vector_only",
            "hybrid_bm25",
            "language_restricted_vector",
            "language_restricted_hybrid",
            "current_linker",
        ]
    ]


def _language_matches(expected_language: str, predicted_concept: str | None) -> bool:
    if predicted_concept is None:
        return False

    language_concepts = LANGUAGE_CONCEPT_IDS.get(expected_language, set())
    return predicted_concept in language_concepts


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


def print_summary(results: list[StrategyResult], gold_path: Path, include_needs_review: bool) -> None:
    print("=== Dry-run Linking Strategy Comparison ===")
    print(f"Gold file: {gold_path}")
    print(f"Target: {'confirmed + needs_review' if include_needs_review else 'confirmed only'}")
    print()
    print(
        f"{'Strategy':<30}"
        f"{'Total':>7}"
        f"{'Exact':>8}"
        f"{'Exact %':>10}"
        f"{'Lang OK':>10}"
        f"{'Lang %':>10}"
        f"{'Missing':>10}"
    )
    print("-" * 85)

    for strategy in [
        "vector_only",
        "hybrid_bm25",
        "language_restricted_vector",
        "language_restricted_hybrid",
        "current_linker",
    ]:
        rows = [row for row in results if row.strategy == strategy]
        total = len(rows)
        exact = sum(row.exact_match for row in rows)
        language_ok = sum(row.predicted_language_match for row in rows)
        missing = sum(row.missing_question or row.missing_candidate for row in rows)
        exact_rate = exact / total * 100 if total else 0
        language_rate = language_ok / total * 100 if total else 0

        print(
            f"{strategy:<30}"
            f"{total:>7}"
            f"{exact:>8}"
            f"{exact_rate:>9.2f}%"
            f"{language_ok:>10}"
            f"{language_rate:>9.2f}%"
            f"{missing:>10}"
        )


def print_details(results: list[StrategyResult], strategy: str) -> None:
    rows = [
        row
        for row in results
        if row.strategy == strategy
        and (not row.exact_match or not row.predicted_language_match)
    ]
    if not rows:
        return

    print()
    print(f"=== Details: {strategy} ===")
    for row in rows:
        print(f"- {row.problem_id}")
        print(f"  expected concept: {row.expected_concept}")
        print(f"  predicted concept: {row.predicted_concept or '-'}")
        print(f"  expected language: {row.expected_language}")
        print(f"  language match: {row.predicted_language_match}")
        if row.final_score is not None:
            print(f"  final/vector/bm25: {row.final_score:.4f} / {row.vector_score:.4f} / {row.bm25_score:.4f}")


def main() -> None:
    args = parse_args()
    gold_path = Path(args.gold)

    cases = load_gold_cases(gold_path, include_needs_review=args.include_needs_review)
    if not cases:
        raise SystemExit("평가할 케이스가 없습니다. gold JSON의 review_status를 확인하세요.")

    graph = create_graph()
    concepts = fetch_concepts(graph)
    ranker = BM25ConceptRanker(concepts)
    questions = fetch_questions(graph, [case.problem_id for case in cases])
    results = predict_all_strategies(
        graph=graph,
        cases=cases,
        questions=questions,
        ranker=ranker,
        vector_top_k=args.vector_top_k,
    )

    print_summary(
        results=results,
        gold_path=gold_path,
        include_needs_review=args.include_needs_review,
    )
    if args.show_details:
        print_details(results=results, strategy=args.details_strategy)


if __name__ == "__main__":
    main()
