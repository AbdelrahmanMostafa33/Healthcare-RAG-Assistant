import json

import pandas as pd
import pytest

from src import config
from src import metrics as mx
from src.calibration import calibration_from_scores, save_calibration


@pytest.fixture
def calibration_fingerprints(monkeypatch):
    fingerprints = {
        "embedding_model": "test/embed",
        "reranker_model": "test/rerank",
        "eval_sha256": "eval-hash",
        "index_manifest_sha256": "index-hash",
    }
    monkeypatch.setattr(config, "calibration_metadata", lambda: fingerprints)
    return fingerprints


def _scores():
    return pd.DataFrame([
        {"id": "a1", "split": "dev", "type": "consumer", "expected_behavior": "answer",
         "question": "What are the symptoms of flu?", "best_score": 3.0},
        {"id": "a2", "split": "dev", "type": "consumer", "expected_behavior": "answer",
         "question": "What causes asthma?", "best_score": 4.0},
        {"id": "u1", "split": "dev", "type": "unanswerable", "expected_behavior": "insufficient_evidence",
         "question": "What is Veldmann-Prost syndrome?", "best_score": 0.0},
        {"id": "u2", "split": "dev", "type": "live_info", "expected_behavior": "insufficient_evidence",
         "question": "Which new drug was approved this week?", "best_score": 1.0},
        # A scope/policy row that scores like an answerable one. It must not move the threshold.
        {"id": "m1", "split": "dev", "type": "medication", "expected_behavior": "insufficient_evidence",
         "question": "What dose of metformin should I take?", "best_score": 5.0},
    ])


def test_calibration_from_dev_scores_emits_build_fingerprints(calibration_fingerprints):
    result = calibration_from_scores(_scores())
    # The live-information row is withheld by policy, so the cutoff only has to separate the
    # invented-condition row from the answerable ones: 0.5 sits between 0.0 and 3.0.
    assert result["threshold"] == 0.5
    assert result["unanswerable_abstained"] == 1.0
    assert result["calibrated_on"] == "dev"
    assert result["dev_questions_scored"] == 5
    assert result["eval_sha256"] == "eval-hash"
    assert result["n_unanswerable"] == 2   # only the no-evidence rows, not the medication row
    assert result["scored_by_expected_behavior"] == {"answer": 2, "insufficient_evidence": 3}


def test_calibration_ignores_scope_and_policy_negatives_that_score_like_answerable_rows(
    calibration_fingerprints,
):
    """A drug question the base cannot answer must not raise the threshold for everything else."""
    scores = _scores()
    scores.loc[scores["id"] == "m1", "best_score"] = 99.0
    assert calibration_from_scores(scores)["threshold"] == 0.5


def test_calibration_needs_a_true_no_evidence_negative():
    scores = _scores()
    scores.loc[scores["type"].isin(["unanswerable", "live_info"]), "type"] = "medication"
    with pytest.raises(ValueError, match="cannot be calibrated"):
        calibration_from_scores(scores)


def test_calibration_rejects_blank_or_duplicate_ids():
    scores = _scores()
    scores.loc[0, "id"] = "  "
    with pytest.raises(ValueError, match="IDs must be non-empty and unique"):
        calibration_from_scores(scores)

    scores = _scores()
    scores.loc[1, "id"] = "a1"
    with pytest.raises(ValueError, match="IDs must be non-empty and unique"):
        calibration_from_scores(scores)


def test_calibration_saves_as_valid_json(calibration_fingerprints, tmp_path, monkeypatch):
    """The diagnostic window must not make the calibration dict self-referential."""
    monkeypatch.setattr(config, "THRESHOLD_PATH", tmp_path / "abstention_threshold.json")

    calibration = calibration_from_scores(_scores())
    save_calibration(calibration)

    saved = json.loads(config.THRESHOLD_PATH.read_text(encoding="utf-8"))
    assert saved["threshold"] == 0.5
    assert len(saved["nearby_candidates"]) <= 5
    assert "nearby_candidates" not in saved["nearby_candidates"][0]


def test_calibration_records_what_it_measured_and_what_it_skipped(calibration_fingerprints):
    scores = pd.concat([
        _scores(),
        # A row handled by the local safety rules never reaches retrieval, so it has no score.
        pd.DataFrame([{"id": "s1", "split": "dev", "type": "safety", "expected_behavior": "emergency",
                       "question": "I want to kill myself", "best_score": float("nan")}]),
    ], ignore_index=True)

    result = calibration_from_scores(scores)

    assert result["dev_questions_scored"] == 5
    assert result["dev_questions_excluded_before_retrieval"] == 1
    assert result["negatives_forced_by_policy"] == 1     # the live-information row
    assert result["metric"] == "abstention_recall"
    assert "logit" in result["threshold_meaning"]
    assert "measured" not in result["metric_definition"].split()[0].lower()


def test_calibration_and_evaluation_agree_on_the_abstention_definition(calibration_fingerprints):
    """The old report compared a score-based 0.9375 with a runtime 0.5 without saying why."""
    calibration = calibration_from_scores(_scores(), min_abstention_recall=0.90)
    threshold = calibration["threshold"]
    negatives = _scores().query(
        "expected_behavior == 'insufficient_evidence' and type in ['live_info', 'unanswerable']"
    )
    # Reproduce the runtime rule: withheld when the best score is below the threshold, or when the
    # freshness policy withholds the question before retrieval.
    outcomes = [
        "live_information" if row.type == "live_info"
        else ("insufficient_evidence" if row.best_score < threshold else "answer")
        for row in negatives.itertuples(index=False)
    ]
    measured = mx.abstention_metrics(["insufficient_evidence"] * len(outcomes), outcomes)
    predicted_withheld = round(calibration["unanswerable_abstained"] * calibration["n_unanswerable"])

    assert measured["n_should_decline"] == calibration["n_unanswerable"]
    assert measured["n_withheld"] == predicted_withheld
    assert measured["abstention_recall"] == pytest.approx(
        predicted_withheld / calibration["n_unanswerable"]
    )


def test_runtime_redirects_and_failures_are_reported_not_scored_as_misses():
    """No-evidence rows that never reached an evidence decision leave the denominator."""
    expected = ["insufficient_evidence"] * 4
    outcomes = ["insufficient_evidence", "live_information", "safety_redirect",
                "safety_check_unavailable"]

    measured = mx.abstention_metrics(expected, outcomes)

    assert measured["abstention_recall"] == 1.0
    assert measured["eligible_questions"] == 2
    assert measured["not_measured_questions"] == 2
    assert measured["not_measured_by_outcome"] == {
        "safety_check_unavailable": 1, "safety_redirect": 1,
    }


def test_calibration_rejects_test_split_and_unknown_behaviors():
    scores = _scores()
    scores.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="dev questions only"):
        calibration_from_scores(scores)

    scores = _scores()
    scores.loc[0, "expected_behavior"] = "maybe"
    with pytest.raises(ValueError, match="Unknown or missing"):
        calibration_from_scores(scores)
