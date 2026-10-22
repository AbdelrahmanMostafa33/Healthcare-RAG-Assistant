import pandas as pd
import pytest

from src import config
from src.calibration import calibration_from_scores


def _scores():
    return pd.DataFrame([
        {"id": "a1", "split": "dev", "type": "consumer", "expected_behavior": "answer", "best_score": 3.0},
        {"id": "a2", "split": "dev", "type": "consumer", "expected_behavior": "answer", "best_score": 4.0},
        {"id": "u1", "split": "dev", "type": "unanswerable", "expected_behavior": "insufficient_evidence", "best_score": 0.0},
        {"id": "u2", "split": "dev", "type": "live_info", "expected_behavior": "insufficient_evidence", "best_score": 1.0},
        # A scope/policy row that scores like an answerable one. It must not move the threshold.
        {"id": "m1", "split": "dev", "type": "medication", "expected_behavior": "insufficient_evidence", "best_score": 5.0},
    ])


def test_calibration_from_dev_scores_emits_build_fingerprints(monkeypatch):
    monkeypatch.setattr(config, "calibration_metadata", lambda: {
        "embedding_model": "test/embed",
        "reranker_model": "test/rerank",
        "eval_sha256": "eval-hash",
        "index_manifest_sha256": "index-hash",
    })
    result = calibration_from_scores(_scores())
    assert result["threshold"] == 2.0
    assert result["unanswerable_abstained"] == 1.0
    assert result["calibrated_on"] == "dev"
    assert result["dev_questions_scored"] == 5
    assert result["eval_sha256"] == "eval-hash"
    assert result["n_unanswerable"] == 2   # only the no-evidence rows, not the medication row
    assert result["scored_by_expected_behavior"] == {"answer": 2, "insufficient_evidence": 3}


def test_calibration_ignores_scope_and_policy_negatives_that_score_like_answerable_rows():
    """A drug question the base cannot answer must not raise the threshold for everything else."""
    scores = _scores()
    scores.loc[scores["id"] == "m1", "best_score"] = 99.0
    assert calibration_from_scores(scores)["threshold"] == 2.0


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


def test_calibration_rejects_test_split_and_unknown_behaviors():
    scores = _scores()
    scores.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="dev questions only"):
        calibration_from_scores(scores)

    scores = _scores()
    scores.loc[0, "expected_behavior"] = "maybe"
    with pytest.raises(ValueError, match="Unknown or missing"):
        calibration_from_scores(scores)
