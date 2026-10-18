"""Small regression checks for the evaluation results that appear in notebook 04."""

import json

import pandas as pd
import pytest

from src import evaluation as ev
from src import metrics as mx


def _citation_result(answer, source_ids, passage_count=5):
    return {
        "answer": answer,
        "sources": [{"n": n} for n in source_ids],
        "passages": [{"chunk_id": i, "text": f"passage {i}"} for i in range(1, passage_count + 1)],
        "passage_count": passage_count,
        "abstained": False,
    }


@pytest.mark.parametrize("answer,source_ids", [
    ("Claim [1].", [1]),
    ("Claim [1, 2].", [1, 2]),
    ("Claim [1-3].", [1, 2, 3]),
])
def test_citation_metrics_use_the_runtime_parser(answer, source_ids):
    result = mx.citation_metrics({"q": _citation_result(answer, source_ids)})
    assert result["valid_reference_rate"] == 1.0
    assert result["answers_with_valid_source_references"] == 1


def test_citation_metrics_do_not_count_years_or_unreturned_sources():
    year = _citation_result("In 2024 this changed [2024]. Fever is common [1].", [1])
    missing = _citation_result("Claim [1, 2].", [1])
    assert mx.citation_metrics({"year": year})["valid_reference_rate"] == 1.0
    assert mx.citation_metrics({"missing": missing})["valid_reference_rate"] == 0.0


def test_citation_metrics_include_fail_closed_rejections():
    metrics = mx.citation_metrics({
        "rejected": {"answer": "No sources could be cited.", "sources": [], "abstained": True,
                     "citation_validation_failed": True},
    })
    assert metrics["generated_answers"] == 0
    assert metrics["citation_validation_rejections"] == 1


def test_insufficient_evidence_rubric_exempts_the_standard_abstention_closing():
    from src.pipeline import ABSTAIN_MESSAGE

    rubric = ev.CHECKS["insufficient_evidence"]
    assert "call your local emergency number" in rubric
    assert "must NOT be treated as giving emergency advice" in rubric
    assert "If you think this could be an emergency" in ABSTAIN_MESSAGE


def test_coverage_reports_fact_and_question_weighted_values_with_denominators():
    facts = pd.DataFrame({"id": ["q1", "q1", "q2"], "score": [1.0, 1.0, 0.0]})
    coverage = mx.coverage_metrics(facts)
    assert coverage["key_fact_coverage"] == pytest.approx(2 / 3)
    assert coverage["key_fact_coverage_question_weighted"] == pytest.approx(0.5)
    assert coverage["facts_graded"] == 3 and coverage["questions_graded"] == 2


def test_abstention_does_not_count_safety_or_citation_blocks():
    assert not mx._retrieval_abstained({"abstained": True, "abstain_reason": "citation_validation_failed"})
    assert mx._retrieval_abstained({"abstained": True, "abstain_reason": "insufficient_evidence"})
    assert not mx._retrieval_abstained({"abstained": True, "abstain_reason": "uncalibrated"})


def test_evaluation_keeps_errors_out_of_quality_denominators():
    frame = pd.DataFrame([
        {"id": "ok", "type": "consumer", "question": "What is flu?", "key_facts": "Virus",
         "expected_behavior": "answer"},
        {"id": "bad", "type": "consumer", "question": "What is measles?", "key_facts": "Virus",
         "expected_behavior": "answer"},
        {"id": "skip", "type": "unanswerable", "question": "What is Zorbo disease?", "key_facts": None,
         "expected_behavior": "insufficient_evidence"},
    ])
    results = {
        "ok": {"answer": "Flu is viral [1].", "sources": [{"n": 1}], "passage_count": 1,
               "abstained": False, "latency_ms": 500, "status": "ok"},
        "bad": {"answer": "", "sources": [], "abstained": False, "status": "error",
                "error": "provider unavailable"},
        "skip": {"answer": "", "sources": [], "abstained": False, "status": "skipped",
                 "error": "batch stopped"},
    }

    class Judge:
        model = "fake-judge"

        def __call__(self, system, user, max_tokens=700):
            payload = json.loads(user)
            return {
                "key_facts": [{"fact": fact, "verdict": "yes"} for fact in payload["key_facts"]],
                "behavior": {"verdict": "yes", "reason": "ok"},
            }

    scored = ev.evaluate_pipeline_results(frame, results, Judge())
    assert scored["requested_questions"] == 3
    assert scored["attempted_questions"] == 2
    assert scored["pipeline_errors"] == 1 and scored["skipped_questions"] == 1
    assert scored["graded_questions"] == 1
    assert scored["coverage"]["answerable_questions_graded"] == 1
    assert scored["latency"] == {"mean_ms": 500.0, "measured_questions": 1}
