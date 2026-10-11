"""Small, reproducible calibration helpers for the evidence-abstention threshold."""
import json
import os
from datetime import date

import numpy as np
import pandas as pd

from src import config, safety
from src.evaluation import choose_threshold


VALID_BEHAVIORS = {"answer", "insufficient_evidence", "emergency", "safety_redirect", "refuse"}


def validate_evaluation_data(questions: pd.DataFrame) -> None:
    """Fail early on malformed labels, missing report fields or exact train/test leakage."""
    required = {"id", "split", "type", "expected_behavior", "question", "key_facts"}
    missing = required - set(questions.columns)
    if missing:
        raise ValueError(f"Evaluation data is missing columns: {sorted(missing)}")
    if questions.empty:
        raise ValueError("Evaluation data is empty")
    ids = questions["id"].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any() or not ids.is_unique:
        raise ValueError("Evaluation question IDs must be non-empty and unique")
    if questions["question"].isna().any() or questions["question"].astype(str).str.strip().eq("").any():
        raise ValueError("Every evaluation row must have a non-empty question")
    if questions["type"].isna().any() or questions["type"].astype(str).str.strip().eq("").any():
        raise ValueError("Every evaluation row must have a non-empty type")
    normalized_questions = questions["question"].astype(str).str.strip().str.casefold()
    if normalized_questions.duplicated().any():
        duplicates = normalized_questions[normalized_questions.duplicated(keep=False)].unique().tolist()
        raise ValueError(f"Evaluation questions must be unique across splits; duplicates include: {duplicates[:3]}")
    if not set(questions["split"].dropna()) <= {"dev", "test"} or questions["split"].isna().any():
        raise ValueError("Evaluation split values must be 'dev' or 'test'")
    unknown = set(questions["expected_behavior"].dropna()) - VALID_BEHAVIORS
    if unknown or questions["expected_behavior"].isna().any():
        raise ValueError(f"Unknown or missing expected_behavior values: {sorted(unknown)}")
    answerable = questions[questions["expected_behavior"] == "answer"]
    facts = answerable["key_facts"].astype("string").str.strip()
    if facts.isna().any() or facts.eq("").any():
        raise ValueError("Every answerable question must have at least one key fact")


def score_dev_questions(retriever, questions: pd.DataFrame) -> pd.DataFrame:
    """Score dev questions without calling the generator or requiring an API key."""
    validate_evaluation_data(questions)
    dev = questions.loc[questions["split"] == "dev"].copy()
    if dev.empty:
        raise ValueError("No rows with split='dev' were found")

    rows = []
    for row in dev.itertuples(index=False):
        if safety.check(row.question):
            score = np.nan
        else:
            passages = retriever.retrieve(row.question)
            score = passages[0]["rerank_score"] if passages else np.nan
        rows.append({"id": row.id, "split": row.split, "expected_behavior": row.expected_behavior,
                     "question": row.question, "best_score": score})
    return pd.DataFrame(rows)


def calibration_from_scores(scores: pd.DataFrame, min_abstention_recall=0.90) -> dict:
    """Select a threshold from precomputed dev scores and attach compatibility fingerprints."""
    required = {"id", "split", "expected_behavior", "best_score"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Scored questions are missing columns: {sorted(missing)}")
    ids = scores["id"].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any() or not ids.is_unique:
        raise ValueError("Scored calibration question IDs must be non-empty and unique")
    if set(scores["split"].dropna()) != {"dev"} or scores["split"].isna().any():
        raise ValueError("The abstention threshold must be calibrated on dev questions only")
    unknown = set(scores["expected_behavior"].dropna()) - VALID_BEHAVIORS
    if unknown or scores["expected_behavior"].isna().any():
        raise ValueError(f"Unknown or missing expected_behavior labels: {sorted(unknown)}")
    scored = scores.dropna(subset=["best_score"])
    answerable = scored.loc[scored["expected_behavior"] == "answer", "best_score"]
    unanswerable = scored.loc[scored["expected_behavior"] == "insufficient_evidence", "best_score"]
    result = choose_threshold(answerable, unanswerable, min_abstention_recall=min_abstention_recall)
    metadata = config.calibration_metadata()
    if any(value is None for value in metadata.values()):
        raise RuntimeError("Cannot save calibration: evaluation file or built index manifest is missing")
    return {
        **result,
        **metadata,
        "rule": (f"lowest threshold keeping abstention recall >= {min_abstention_recall:.2f} "
                 "on the dev split; threshold is frozen before final test evaluation"),
        "calibrated_on": "dev",
        "dev_questions_scored": int(len(scored)),
        "dev_question_ids": scored["id"].astype(str).tolist(),
        "date": str(date.today()),
    }


def calibrate_threshold(retriever, questions: pd.DataFrame, min_abstention_recall=0.90) -> dict:
    """Score the dev split and pick its lowest threshold meeting the abstention-recall floor."""
    return calibration_from_scores(
        score_dev_questions(retriever, questions), min_abstention_recall=min_abstention_recall
    )


def save_calibration(calibration: dict) -> None:
    """Persist a completed calibration atomically so partial writes cannot be trusted."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    temporary = config.THRESHOLD_PATH.with_name(config.THRESHOLD_PATH.name + ".tmp")
    try:
        temporary.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, config.THRESHOLD_PATH)
    finally:
        if temporary.exists():
            temporary.unlink()
