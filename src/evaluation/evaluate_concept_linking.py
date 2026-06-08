import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph


DEFAULT_GOLD_PATH = Path("data/evaluation/question_concept_gold.seed.json")


@dataclass
class GoldCase:
    problem_id: str
    primary_concept: str
    labels: list[str]
    review_status: str
    note: str


@dataclass
class Prediction:
    problem_id: str
    question_exists: bool
    predicted_concepts: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="평가셋 기준으로 Question-Concept VERIFIED_MENTIONS 연결 정확도를 측정합니다."
    )
    parser.add_argument(
        "--gold",
        default=str(DEFAULT_GOLD_PATH),
        help="평가셋 JSON 경로입니다. 기본값: data/evaluation/question_concept_gold.seed.json",
    )
    parser.add_argument(
        "--include-needs-review",
        action="store_true",
        help="review_status가 needs_review인 샘플도 평가에 포함합니다.",
    )
    parser.add_argument(
        "--show-correct",
        action="store_true",
        help="정답 처리된 케이스까지 상세 목록에 출력합니다.",
    )
    return parser.parse_args()


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
                labels=[str(value) for value in item.get("labels", [])],
                review_status=review_status,
                note=str(item.get("note", "")),
            )
        )

    return cases


def create_graph() -> Neo4jGraph:
    load_dotenv()

    return Neo4jGraph(
        url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
    )


def fetch_predictions(graph: Neo4jGraph, problem_ids: list[str]) -> dict[str, Prediction]:
    # UNWIND를 쓰면 평가셋 순서를 유지하면서 누락된 Question도 함께 확인할 수 있습니다.
    query = """
    UNWIND $problem_ids AS problem_id
    OPTIONAL MATCH (q:Question {problem_id: problem_id})
    OPTIONAL MATCH (q)-[:VERIFIED_MENTIONS]->(c:Concept)
    RETURN
        problem_id,
        q IS NOT NULL AS question_exists,
        [concept_id IN collect(c.section_id) WHERE concept_id IS NOT NULL] AS predicted_concepts
    """
    rows = graph.query(query, {"problem_ids": problem_ids})

    return {
        row["problem_id"]: Prediction(
            problem_id=row["problem_id"],
            question_exists=bool(row["question_exists"]),
            predicted_concepts=[str(value) for value in row["predicted_concepts"]],
        )
        for row in rows
    }


def evaluate(cases: list[GoldCase], predictions: dict[str, Prediction]) -> dict:
    results = []
    counts = {
        "total": len(cases),
        "correct": 0,
        "wrong": 0,
        "missing_link": 0,
        "missing_question": 0,
    }

    for case in cases:
        prediction = predictions.get(
            case.problem_id,
            Prediction(case.problem_id, question_exists=False, predicted_concepts=[]),
        )
        predicted_set = set(prediction.predicted_concepts)

        if not prediction.question_exists:
            status = "missing_question"
        elif not prediction.predicted_concepts:
            status = "missing_link"
        elif case.primary_concept in predicted_set:
            status = "correct"
        else:
            status = "wrong"

        counts[status] += 1
        results.append(
            {
                "status": status,
                "problem_id": case.problem_id,
                "expected": case.primary_concept,
                "predicted": prediction.predicted_concepts,
                "labels": case.labels,
                "review_status": case.review_status,
                "note": case.note,
            }
        )

    return {"counts": counts, "results": results}


def print_summary(gold_path: Path, include_needs_review: bool, evaluation: dict, show_correct: bool) -> None:
    counts = evaluation["counts"]
    total = counts["total"]
    accuracy = counts["correct"] / total * 100 if total else 0

    print("=== VERIFIED_MENTIONS baseline ===")
    print(f"Gold file: {gold_path}")
    print(f"Target: {'confirmed + needs_review' if include_needs_review else 'confirmed only'}")
    print(f"Total: {total}")
    print(f"Correct: {counts['correct']}")
    print(f"Wrong: {counts['wrong']}")
    print(f"Missing link: {counts['missing_link']}")
    print(f"Missing question: {counts['missing_question']}")
    print(f"Accuracy: {accuracy:.2f}%")

    detail_statuses = {"wrong", "missing_link", "missing_question"}
    if show_correct:
        detail_statuses.add("correct")

    details = [row for row in evaluation["results"] if row["status"] in detail_statuses]
    if not details:
        return

    print("\n=== Details ===")
    for row in details:
        print(f"- [{row['status']}] {row['problem_id']}")
        print(f"  expected: {row['expected']}")
        print(f"  predicted: {', '.join(row['predicted']) if row['predicted'] else '-'}")
        print(f"  labels: {', '.join(row['labels']) if row['labels'] else '-'}")
        if row["note"]:
            print(f"  note: {row['note']}")


def main() -> None:
    args = parse_args()
    gold_path = Path(args.gold)

    cases = load_gold_cases(gold_path, include_needs_review=args.include_needs_review)
    if not cases:
        raise SystemExit("평가할 케이스가 없습니다. gold JSON의 review_status를 확인하세요.")

    graph = create_graph()
    predictions = fetch_predictions(graph, [case.problem_id for case in cases])
    evaluation = evaluate(cases, predictions)
    print_summary(
        gold_path=gold_path,
        include_needs_review=args.include_needs_review,
        evaluation=evaluation,
        show_correct=args.show_correct,
    )


if __name__ == "__main__":
    main()
