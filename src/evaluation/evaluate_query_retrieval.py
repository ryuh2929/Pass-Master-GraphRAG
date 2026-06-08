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

from src.ingestion.hybrid_linker import BM25ConceptRanker
from src.retrieval.embedder import TEIEmbedder


DEFAULT_GOLD_PATH = Path("data/evaluation/query_retrieval_gold.seed.json")
DEFAULT_WEIGHTS = {
    "hybrid_80_20": (0.8, 0.2),
    "hybrid_60_40": (0.6, 0.4),
    "hybrid_40_60": (0.4, 0.6),
}


@dataclass
class GoldCase:
    case_id: str
    query: str
    primary_concept: str
    acceptable_concepts: set[str]
    intent: str
    difficulty: str
    note: str


@dataclass
class RankedCandidate:
    section_id: str
    title: str
    final_score: float
    vector_score: float
    bm25_score: float


@dataclass
class CaseResult:
    strategy: str
    case: GoldCase
    candidates: list[RankedCandidate]

    @property
    def predicted(self) -> str | None:
        return self.candidates[0].section_id if self.candidates else None

    @property
    def rank(self) -> int | None:
        for index, candidate in enumerate(self.candidates, start=1):
            if _normalize_concept_id(candidate.section_id) in self.case.acceptable_concepts:
                return index
        return None

    def hit_at(self, k: int) -> bool:
        return self.rank is not None and self.rank <= k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LLM query refine 없이 raw query로 Concept retrieval 정확도를 평가합니다. "
            "OpenAI/LLM 호출은 하지 않습니다."
        )
    )
    parser.add_argument(
        "--gold",
        default=str(DEFAULT_GOLD_PATH),
        help="Query retrieval gold JSON 경로입니다.",
    )
    parser.add_argument(
        "--vector-top-k",
        type=int,
        default=int(os.getenv("VECTOR_TOP_K", "100")),
        help="Neo4j vector index에서 가져올 후보 수입니다.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="전략별 최종 후보를 몇 개까지 보관/평가할지 정합니다.",
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="오답 또는 top-1 실패 케이스 상세를 출력합니다.",
    )
    parser.add_argument(
        "--details-strategy",
        default="hybrid_60_40",
        choices=["vector_only", "bm25_only", *DEFAULT_WEIGHTS.keys()],
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


def load_gold_cases(gold_path: Path) -> list[GoldCase]:
    with gold_path.open("r", encoding="utf-8") as file:
        raw_cases = json.load(file)

    cases = []
    for item in raw_cases:
        acceptable = {
            _normalize_concept_id(concept_id)
            for concept_id in item.get("acceptable_concepts") or [item["primary_concept"]]
        }
        cases.append(
            GoldCase(
                case_id=str(item["case_id"]),
                query=str(item["query"]),
                primary_concept=_normalize_concept_id(item["primary_concept"]),
                acceptable_concepts=acceptable,
                intent=str(item.get("intent", "")),
                difficulty=str(item.get("difficulty", "")),
                note=str(item.get("note", "")),
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


def fetch_vector_scores(
    graph: Neo4jGraph,
    query_vector: list[float],
    top_k: int,
) -> dict[str, float]:
    rows = graph.query(
        """
        CALL db.index.vector.queryNodes('concept_index', $top_k, $vector)
        YIELD node AS c, score
        RETURN c.section_id AS section_id, score
        """,
        {"top_k": top_k, "vector": query_vector},
    )
    return {
        _normalize_concept_id(row["section_id"]): float(row["score"])
        for row in rows
        if row["section_id"] is not None and row["score"] >= 0
    }


def evaluate_cases(
    *,
    graph: Neo4jGraph,
    embedder: TEIEmbedder,
    ranker: BM25ConceptRanker,
    cases: list[GoldCase],
    vector_top_k: int,
    final_top_k: int,
) -> list[CaseResult]:
    results = []

    for index, case in enumerate(cases, start=1):
        query_vector = embedder.get_embedding(case.query)
        vector_scores = fetch_vector_scores(graph, query_vector, top_k=vector_top_k)
        bm25_scores = {
            _normalize_concept_id(section_id): score
            for section_id, score in ranker.get_scores(case.query).items()
        }

        strategies = {
            "vector_only": build_ranked_candidates(
                ranker=ranker,
                vector_scores=vector_scores,
                bm25_scores={},
                vector_weight=1.0,
                bm25_weight=0.0,
                top_k=final_top_k,
            ),
            "bm25_only": build_ranked_candidates(
                ranker=ranker,
                vector_scores={},
                bm25_scores=bm25_scores,
                vector_weight=0.0,
                bm25_weight=1.0,
                top_k=final_top_k,
            ),
        }

        for strategy, (vector_weight, bm25_weight) in DEFAULT_WEIGHTS.items():
            strategies[strategy] = build_ranked_candidates(
                ranker=ranker,
                vector_scores=vector_scores,
                bm25_scores=bm25_scores,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                top_k=final_top_k,
            )

        for strategy, candidates in strategies.items():
            results.append(CaseResult(strategy=strategy, case=case, candidates=candidates))

        if index % 10 == 0:
            print(f"Evaluated query cases: {index}/{len(cases)}")

    return results


def build_ranked_candidates(
    *,
    ranker: BM25ConceptRanker,
    vector_scores: dict[str, float],
    bm25_scores: dict[str, float],
    vector_weight: float,
    bm25_weight: float,
    top_k: int,
) -> list[RankedCandidate]:
    candidate_ids = set(vector_scores) | set(bm25_scores)
    normalized_vector = normalize_scores(vector_scores)
    normalized_bm25 = normalize_scores(bm25_scores)

    candidates = []
    for section_id in candidate_ids:
        concept = get_ranker_concept(ranker, section_id)
        if not concept:
            continue

        final_score = (
            vector_weight * normalized_vector.get(section_id, 0.0)
            + bm25_weight * normalized_bm25.get(section_id, 0.0)
        )
        candidates.append(
            RankedCandidate(
                section_id=_normalize_concept_id(concept.section_id),
                title=concept.title,
                final_score=final_score,
                vector_score=vector_scores.get(section_id, 0.0),
                bm25_score=bm25_scores.get(section_id, 0.0),
            )
        )

    return sorted(candidates, key=lambda candidate: candidate.final_score, reverse=True)[:top_k]


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
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


def get_ranker_concept(ranker: BM25ConceptRanker, section_id: str):
    normalized_id = _normalize_concept_id(section_id)
    for concept in ranker.concepts:
        if _normalize_concept_id(concept.section_id) == normalized_id:
            return concept
    return None


def print_summary(results: list[CaseResult], gold_path: Path) -> None:
    print("=== Query Retrieval Evaluation (No LLM Refine) ===")
    print(f"Gold file: {gold_path}")
    print()
    print(
        f"{'Strategy':<18}"
        f"{'Total':>7}"
        f"{'Top1':>8}"
        f"{'Top1 %':>10}"
        f"{'Top3':>8}"
        f"{'Top3 %':>10}"
        f"{'Top5':>8}"
        f"{'Top5 %':>10}"
        f"{'MRR':>8}"
    )
    print("-" * 87)

    for strategy in ["vector_only", "bm25_only", *DEFAULT_WEIGHTS.keys()]:
        rows = [row for row in results if row.strategy == strategy]
        total = len(rows)
        top1 = sum(row.hit_at(1) for row in rows)
        top3 = sum(row.hit_at(3) for row in rows)
        top5 = sum(row.hit_at(5) for row in rows)
        mrr = sum((1 / row.rank) for row in rows if row.rank) / total if total else 0.0
        print(
            f"{strategy:<18}"
            f"{total:>7}"
            f"{top1:>8}"
            f"{_rate(top1, total):>9.2f}%"
            f"{top3:>8}"
            f"{_rate(top3, total):>9.2f}%"
            f"{top5:>8}"
            f"{_rate(top5, total):>9.2f}%"
            f"{mrr:>8.3f}"
        )


def print_breakdown(results: list[CaseResult], strategy: str) -> None:
    rows = [row for row in results if row.strategy == strategy]
    print()
    print(f"=== Breakdown: {strategy} ===")
    for field_name in ["intent", "difficulty"]:
        values = sorted({getattr(row.case, field_name) for row in rows if getattr(row.case, field_name)})
        print(f"\n[{field_name}]")
        for value in values:
            subset = [row for row in rows if getattr(row.case, field_name) == value]
            top1 = sum(row.hit_at(1) for row in subset)
            top3 = sum(row.hit_at(3) for row in subset)
            print(
                f"- {value:<16} total={len(subset):>2} "
                f"top1={_rate(top1, len(subset)):>6.2f}% "
                f"top3={_rate(top3, len(subset)):>6.2f}%"
            )


def print_details(results: list[CaseResult], strategy: str) -> None:
    rows = [
        row
        for row in results
        if row.strategy == strategy and not row.hit_at(1)
    ]
    if not rows:
        return

    print()
    print(f"=== Details: {strategy} top-1 failures ===")
    for row in rows:
        print(f"- {row.case.case_id}: {row.case.query}")
        print(f"  expected: {sorted(row.case.acceptable_concepts)}")
        print(f"  predicted: {row.predicted or '-'}")
        print(f"  rank: {row.rank or '-'}")
        for candidate in row.candidates[:5]:
            print(
                "  "
                f"{candidate.section_id:<3} "
                f"final/vector/bm25="
                f"{candidate.final_score:.4f}/"
                f"{candidate.vector_score:.4f}/"
                f"{candidate.bm25_score:.4f} "
                f"{candidate.title}"
            )
        if row.case.note:
            print(f"  note: {row.case.note}")


def _normalize_concept_id(value) -> str:
    text = str(value).strip()
    return text.zfill(3) if text.isdigit() else text


def _rate(count: int, total: int) -> float:
    return count / total * 100 if total else 0.0


def main() -> None:
    args = parse_args()
    gold_path = Path(args.gold)
    cases = load_gold_cases(gold_path)
    if not cases:
        raise SystemExit("평가할 query retrieval 케이스가 없습니다.")

    graph = create_graph()
    concepts = fetch_concepts(graph)
    if not concepts:
        raise SystemExit("Neo4j에서 Concept 노드를 찾지 못했습니다.")

    embedder = TEIEmbedder()
    ranker = BM25ConceptRanker(concepts)
    results = evaluate_cases(
        graph=graph,
        embedder=embedder,
        ranker=ranker,
        cases=cases,
        vector_top_k=args.vector_top_k,
        final_top_k=args.top_k,
    )

    print_summary(results, gold_path)
    print_breakdown(results, strategy=args.details_strategy)
    if args.show_details:
        print_details(results, strategy=args.details_strategy)


if __name__ == "__main__":
    main()
