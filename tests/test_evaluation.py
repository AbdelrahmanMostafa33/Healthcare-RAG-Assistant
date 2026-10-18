import json
import types

import numpy as np
import pandas as pd
import pytest

from src import config
from src import evaluation as ev
from src import metrics as mx
from src.calibration import validate_evaluation_data

# --- threshold -------------------------------------------------------------


def test_choose_threshold_separates_clean_data_with_a_margin():
    best = ev.choose_threshold([5.0, 6.0, 7.0, 4.0], [-3.0, -1.0, 0.5])
    assert best["balanced_accuracy"] == 1.0
    assert 0.5 < best["threshold"] <= 4.0
    assert best["answerable_answered"] == 1.0 and best["unanswerable_abstained"] == 1.0


def test_choose_threshold_takes_the_lowest_value_that_keeps_the_abstention_floor():
    # one unanswerable question scores high (4.5), one answerable scores low (-1)
    best = ev.choose_threshold([6.0, 5.0, 4.0, -1.0], [-4.0, -3.0, -2.0, 4.5])
    # t = -1.5 would answer 4/4 but abstain on only 3/4 (0.75 < 0.90), so the floor forces t = 4.75
    assert best["threshold"] == pytest.approx(4.75)
    assert best["unanswerable_abstained"] == 1.0 and best["answerable_answered"] == 0.5
    assert best["balanced_accuracy"] == pytest.approx(0.75)
    assert best["n_answerable"] == 4 and best["n_unanswerable"] == 4


def test_choose_threshold_lowers_the_threshold_when_the_floor_allows_it():
    scores_unanswerable = list(np.linspace(0, 1, 10)) + [5.0]     # 1 of 11 is high: 10/11 = 0.91 abstained is enough
    best = ev.choose_threshold([6.0, 7.0, 8.0], scores_unanswerable, min_abstention_recall=0.90)
    assert 1.0 < best["threshold"] < 5.0 and best["answerable_answered"] == 1.0
    assert best["unanswerable_abstained"] >= 0.90


def test_choose_threshold_when_the_scores_do_not_separate_the_groups_picks_the_strictest_value():
    # answerable questions score below the unanswerable ones, so only "abstain on everything" meets the floor
    best = ev.choose_threshold([0.0, 1.0], [2.0, 3.0])
    assert best["threshold"] == 4.0 and best["answerable_answered"] == 0.0


def test_choose_threshold_needs_both_groups():
    with pytest.raises(ValueError):
        ev.choose_threshold([], [1.0])


def test_choose_threshold_rejects_questions_that_never_reached_retrieval():
    with pytest.raises(ValueError, match="NaN"):
        ev.choose_threshold([5.0, float("nan")], [1.0, 2.0])


def test_rate_limit_classifier_retries_minute_caps_but_not_daily_quota_exhaustion():
    from src import retry

    minute_limit = types.SimpleNamespace(body={"error": {"message": "Rate limit reached for tokens per minute (TPM)", "type": "tokens"}})
    daily_limit = types.SimpleNamespace(body={"error": {"message": "Tokens per day (TPD) limit reached", "type": "tokens"}})
    quota_limit = types.SimpleNamespace(body={"error": {"message": "You exceeded your quota", "type": "insufficient_quota"}})
    assert not retry.is_hard_quota_exhaustion(minute_limit)
    assert retry.is_hard_quota_exhaustion(daily_limit)
    assert retry.is_hard_quota_exhaustion(quota_limit)


def test_evaluation_dataset_passes_the_shared_data_validator():
    validate_evaluation_data(pd.read_csv(config.EVAL_PATH))


def test_evaluation_data_validator_rejects_exact_question_leakage_between_splits():
    duplicated = pd.DataFrame([
        {"id": "q1", "split": "dev", "type": "consumer", "expected_behavior": "answer", "question": "What is flu?", "key_facts": "fact"},
        {"id": "q2", "split": "test", "type": "consumer", "expected_behavior": "answer", "question": " what is Flu? ", "key_facts": "fact"},
    ])
    with pytest.raises(ValueError, match="unique across splits"):
        validate_evaluation_data(duplicated)


def test_evaluation_data_validator_rejects_duplicate_ids_and_missing_answer_key_facts():
    bad = pd.DataFrame([
        {"id": "q1", "split": "dev", "type": "consumer", "expected_behavior": "answer", "question": "Q?", "key_facts": "fact"},
        {"id": "q1", "split": "test", "type": "consumer", "expected_behavior": "answer", "question": "Q2?", "key_facts": None},
    ])
    with pytest.raises(ValueError, match="unique"):
        validate_evaluation_data(bad)
    bad.loc[1, "id"] = "q2"
    with pytest.raises(ValueError, match="key fact"):
        validate_evaluation_data(bad)

# --- abstention metrics ----------------------------------------------------


def test_abstention_metrics():
    expected = ["answer", "answer", "answer", "insufficient_evidence", "insufficient_evidence", "emergency"]
    declined = [False, True, False, True, False, False]
    m = mx.abstention_metrics(expected, declined)
    assert m["abstention_recall"] == pytest.approx(0.5)       # 1 of 2 unanswerable declined
    assert m["abstention_precision"] == pytest.approx(0.5)    # 1 of 2 declined questions should have been
    assert m["over_refusal"] == pytest.approx(1 / 3)          # 1 of 3 answerable declined


def test_abstention_metrics_handle_missing_denominators():
    m = mx.abstention_metrics(["answer", "answer"], [False, False])
    assert m["abstention_recall"] is None and m["abstention_precision"] is None and m["over_refusal"] == 0.0


def test_abstention_metrics_report_the_scope_and_policy_slice_separately():
    """Drug and vague-symptom rows are not what the threshold is calibrated against."""
    expected = ["answer", "insufficient_evidence", "insufficient_evidence"]
    types = ["consumer", "unanswerable", "medication"]
    m = mx.abstention_metrics(expected, [False, True, False], types)
    assert m["abstention_recall"] == pytest.approx(0.5)
    assert m["policy_scope_abstention"] == 0.0 and m["n_policy_scope"] == 1


def test_a_citation_rejection_is_not_a_retrieval_abstention():
    """Counting it as one turned a single rejected answer into over_refusal = 1.0."""
    assert mx._retrieval_abstained({"abstained": True, "abstain_reason": "citation_validation_failed"}) is False
    assert mx._retrieval_abstained({"abstained": True, "abstain_reason": "insufficient_evidence"}) is True
    assert mx._retrieval_abstained({"abstained": True, "abstain_reason": "uncalibrated"}) is False
    # results saved before abstain_reason existed still behave sensibly
    assert mx._retrieval_abstained({"abstained": True, "citation_validation_failed": True}) is False
    assert mx._retrieval_abstained({"abstained": True}) is True


def test_wilson_interval_stays_inside_zero_and_one_for_small_samples():
    assert mx.wilson_interval(16, 16) == [0.8064, 1.0]
    assert mx.wilson_interval(0, 16) == [0.0, 0.1936]
    assert mx.wilson_interval(0, 0) is None

# --- judge and scoring -----------------------------------------------------


class FakeJudge:
    """Answers each judge prompt from a canned table, keyed by the system prompt used."""
    model = "fake-judge"

    def __init__(self, responses):
        self.responses, self.calls = responses, []

    def __call__(self, system, user, max_tokens=700):
        self.calls.append(system)
        return self.responses[system]

QUESTIONS = pd.DataFrame([
    {"id": "q1", "type": "consumer", "question": "What is flu?", "key_facts": "Caused by a virus | Spreads easily",
     "expected_behavior": "answer"},
    {"id": "q2", "type": "unanswerable", "question": "What is Zorbo disease?", "key_facts": None,
     "expected_behavior": "insufficient_evidence"},
    {"id": "q3", "type": "medication", "question": "Can I take X?", "key_facts": "Ask a doctor",
     "expected_behavior": "safety_redirect"},
])


def test_score_answers_grades_facts_and_behavior_with_one_request_per_question():
    class SequentialJudge:
        model = "fake-judge"

        def __init__(self):
            self.calls = []
            self.responses = [
                {"key_facts": [{"fact": "Spreads easily", "verdict": "partial"},
                                {"fact": "Caused by a virus", "verdict": "yes"}],
                 "behavior": {"verdict": "yes", "reason": "answered"}},
                {"key_facts": [],
                 "behavior": {"verdict": "yes", "reason": "declined unsupported question"}},
                {"key_facts": [],
                 "behavior": {"verdict": "yes", "reason": "redirected safely"}},
            ]

        def __call__(self, system, user, max_tokens=700):
            self.calls.append((system, json.loads(user)))
            return self.responses.pop(0)

    judge = SequentialJudge()
    answers = {"q1": "Flu is caused by a virus.", "q2": "I could not verify that.", "q3": "Ask your doctor."}
    facts, behavior, grading = ev.score_answers(judge, QUESTIONS, answers)
    assert len(judge.calls) == len(QUESTIONS)
    assert all(system == ev.ANSWER_SCORE_SYSTEM for system, _ in judge.calls)
    assert facts["id"].tolist() == ["q1", "q1"]
    assert facts["score"].tolist() == [1.0, 0.5]
    assert facts["judge_model"].unique().tolist() == ["fake-judge"]
    assert behavior["expected"].tolist() == ["answer", "insufficient_evidence", "safety_redirect"]
    assert behavior["ok"].tolist() == [True, True, True]
    assert grading["status"].tolist() == ["ok", "ok", "ok"]
    assert judge.calls[2][1]["behavior_check"] == ev.CHECKS["safety_redirect"]


def test_malformed_judge_output_is_not_counted_as_an_answer_failure():
    class Judge:
        model = "fake-judge"

        def __init__(self):
            self.calls = 0

        def __call__(self, system, user, max_tokens=700):
            self.calls += 1
            return {"key_facts": "invalid", "behavior": {"verdict": "maybe"}}

    facts, behavior, grading = ev.score_answers(Judge(), QUESTIONS.head(1), {"q1": "Flu."})
    assert facts.empty and behavior.empty
    assert grading["status"].tolist() == ["error"]
    assert grading.loc[0, "error"].startswith("ValueError: ")


def test_one_ungradeable_row_does_not_discard_every_verdict_after_it():
    """A malformed grade used to stop the judge loop, blanking the whole report."""

    class Judge:
        model = "fake-judge"

        def __init__(self):
            self.calls = 0

        def __call__(self, system, user, max_tokens=700):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("judge returned an invalid key-fact verdict")
            payload = json.loads(user)
            return {
                "key_facts": [{"fact": fact, "verdict": "yes"} for fact in payload["key_facts"]],
                "behavior": {"verdict": "yes", "reason": "ok"},
            }

    facts, behavior, grading = ev.score_answers(Judge(), QUESTIONS, {"q1": "a", "q2": "b", "q3": "c"})
    assert grading["status"].tolist() == ["error", "ok", "ok"]
    assert behavior["id"].tolist() == ["q2", "q3"]
    assert grading.loc[0, "error"].startswith("ValueError: ")


def test_echoed_key_facts_match_ignoring_case_spacing_and_trailing_punctuation():
    """The judge echoes fact strings; a stray period must not throw away a valid grade."""

    class Judge:
        model = "fake-judge"

        def __call__(self, system, user, max_tokens=700):
            return {
                "key_facts": [
                    {"fact": "  Caused By A Virus. ", "verdict": "yes"},
                    {"fact": "spreads\n  easily", "verdict": "partial"},
                ],
                "behavior": {"verdict": "yes", "reason": "answered"},
            }

    facts, behavior, grading = ev.score_answers(Judge(), QUESTIONS.head(1), {"q1": "Flu."})
    assert grading["status"].tolist() == ["ok"]
    assert facts["fact"].tolist() == ["Caused by a virus", "Spreads easily"]
    assert facts["score"].tolist() == [1.0, 0.5]


def test_a_key_fact_that_is_not_the_requested_one_is_still_an_error():
    """Tolerating punctuation must not tolerate grading a different fact."""

    class Judge:
        model = "fake-judge"

        def __call__(self, system, user, max_tokens=700):
            return {
                "key_facts": [
                    {"fact": "Caused by a virus", "verdict": "yes"},
                    {"fact": "Caused by a bacterium", "verdict": "yes"},
                ],
                "behavior": {"verdict": "yes", "reason": "answered"},
            }

    facts, behavior, grading = ev.score_answers(Judge(), QUESTIONS.head(1), {"q1": "Flu."})
    assert facts.empty and behavior.empty
    assert grading["status"].tolist() == ["error"]


def test_score_answers_preserves_partial_grades_when_judge_quota_is_exhausted():
    from src.retry import ProviderQuotaExceeded

    class Judge:
        model = "fake-judge"

        def __init__(self):
            self.calls = 0

        def __call__(self, system, user, max_tokens=700):
            self.calls += 1
            if self.calls == 2:
                raise ProviderQuotaExceeded("private quota detail")
            payload = json.loads(user)
            return {
                "key_facts": [{"fact": fact, "verdict": "yes"} for fact in payload["key_facts"]],
                "behavior": {"verdict": "yes", "reason": "ok"},
            }

    judge = Judge()
    facts, behavior, grading = ev.score_answers(judge, QUESTIONS, {"q1": "a", "q2": "b", "q3": "c"})
    assert judge.calls == 2
    assert grading["status"].tolist() == ["ok", "error", "skipped"]
    assert grading.loc[1, "error"] == "provider quota exhausted"
    assert facts["id"].unique().tolist() == ["q1"]
    assert behavior["id"].tolist() == ["q1"]
    assert "private quota detail" not in str(grading.to_dict())


def reply(content):
    message = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message, finish_reason="stop")])


class ScriptedClient:
    def __init__(self, *contents):
        self.contents, self.calls = list(contents), []
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return reply(self.contents.pop(0))


def test_the_judge_parses_json_and_uses_the_configured_model():
    client = ScriptedClient('{"verdict": "yes"}')
    judge = ev.make_judge(client)
    assert judge("system", "user") == {"verdict": "yes"}
    assert judge.model == config.JUDGE_MODEL and client.calls[0]["model"] == config.JUDGE_MODEL
    assert client.calls[0]["temperature"] == 0.0 and client.calls[0]["response_format"] == {"type": "json_object"}
    assert client.calls[0]["max_completion_tokens"] == 700
    assert client.calls[0]["reasoning_effort"] == "low"


def test_the_judge_does_not_spend_extra_calls_retrying_invalid_json():
    client = ScriptedClient("not json")
    with pytest.raises(RuntimeError, match="invalid JSON"):
        ev.make_judge(client)("s", "u")
    assert len(client.calls) == 1


def test_the_judge_invalid_json_error_does_not_echo_model_output():
    with pytest.raises(RuntimeError, match="invalid JSON") as error:
        ev.make_judge(ScriptedClient("private generated text"))("s", "u")
    assert "private generated text" not in str(error.value)


def test_make_judge_client_builds_groq_client_with_sdk_retries_disabled(monkeypatch, caplog):
    import src.provider_clients as providers
    from types import SimpleNamespace

    seen = {}

    def fake_openai(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **call: call),
        ))

    monkeypatch.setenv("GROQ_API_KEY", "groq-single")
    monkeypatch.setattr(providers, "OpenAI", fake_openai)
    client = ev.make_judge_client()
    assert hasattr(client, "chat") and hasattr(client.chat, "completions")
    assert client.chat.completions.create(model="test-model") == {"model": "test-model"}
    assert seen["api_key"] == "groq-single"
    assert seen["max_retries"] == 0 and str(seen["base_url"]).startswith(config.GROQ_BASE_URL)
    assert "groq-single" not in caplog.text


def test_citation_metrics_report_valid_refs_and_fail_closed_rejections():
    metrics = mx.citation_metrics({
        "good": {"answer": "Statement [1].", "sources": [{"n": 1}], "abstained": False},
        "bad": {"answer": "I couldn't verify the answer.", "sources": [], "abstained": True,
                "citation_validation_failed": True},
        "no-ref": {"answer": "Statement without a citation.", "sources": [{"n": 1}], "abstained": False},
    })
    assert metrics["generated_answers"] == 2
    assert metrics["answers_with_valid_source_references"] == 1
    assert metrics["valid_reference_rate"] == pytest.approx(0.5)
    assert metrics["citation_validation_rejections"] == 1


def test_citation_metrics_accept_non_contiguous_prompt_citation_ids():
    metrics = mx.citation_metrics({
        "answer": {"answer": "Claim [4].", "sources": [{"n": 4, "title": "Source"}], "abstained": False},
    })
    assert metrics["valid_reference_rate"] == 1.0


def test_pipeline_errors_are_not_graded_as_empty_answers_or_abstentions():
    df = pd.DataFrame([
        {"id": "bad", "type": "consumer", "question": "What is measles?", "key_facts": "Virus",
         "expected_behavior": "answer"},
        {"id": "ok", "type": "consumer", "question": "What is flu?", "key_facts": "Virus",
         "expected_behavior": "answer"},
    ])
    results = {
        "ok": {"answer": "Flu is viral [1].", "sources": [{"n": 1}], "abstained": False,
               "latency_ms": 10, "status": "ok"},
        "bad": {"answer": "", "sources": [], "abstained": False, "error": "provider unavailable",
                "latency_ms": None, "status": "error"},
    }
    judge = FakeJudge({
        ev.ANSWER_SCORE_SYSTEM: {"key_facts": [{"fact": "Virus", "verdict": "yes"}],
                                 "behavior": {"verdict": "yes", "reason": "answered"}},
    })
    scored = ev.evaluate_pipeline_results(df, results, judge)
    assert scored["pipeline_errors"] == 1 and scored["graded_questions"] == 1
    assert scored["requested_questions"] == 2 and scored["attempted_questions"] == 2
    assert scored["graded_questions"] == 1 and scored["grading_errors"] == 0
    assert scored["ungraded_questions"] == 0
    assert scored["facts"]["id"].tolist() == ["ok"]
    assert scored["behavior"]["id"].tolist() == ["ok"]
    assert scored["abstention"]["over_refusal"] == 0.0
    assert len(judge.calls) == 1


def test_skipped_questions_are_excluded_from_quality_denominators():
    df = QUESTIONS.copy()
    results = {
        "q1": {"answer": "Flu is viral [1].", "sources": [{"n": 1}], "abstained": False,
               "latency_ms": 10, "status": "ok"},
        "q2": {"answer": "", "sources": [], "abstained": False, "error": "batch stopped",
               "latency_ms": None, "status": "skipped"},
        "q3": {"answer": "", "sources": [], "abstained": False, "error": "batch stopped",
               "latency_ms": None, "status": "skipped"},
    }
    judge = FakeJudge({
        ev.ANSWER_SCORE_SYSTEM: {"key_facts": [{"fact": "Caused by a virus", "verdict": "yes"},
                                                  {"fact": "Spreads easily", "verdict": "yes"}],
                                 "behavior": {"verdict": "yes", "reason": "ok"}},
    })
    scored = ev.evaluate_pipeline_results(df, results, judge)
    assert scored["requested_questions"] == 3
    assert scored["attempted_questions"] == 1
    assert scored["graded_questions"] == 1
    assert scored["skipped_questions"] == 2
    assert scored["pipeline_errors"] == 0
    assert scored["graded_questions"] == 1 and scored["grading_errors"] == 0
    assert scored["ungraded_questions"] == 0
    assert scored["behavior"]["id"].tolist() == ["q1"]


def test_partial_judge_coverage_is_separate_from_pipeline_coverage():
    from src.retry import ProviderQuotaExceeded

    results = {
        row.id: {"answer": "A response.", "sources": [], "abstained": False,
                 "latency_ms": 10, "status": "ok"}
        for row in QUESTIONS.itertuples()
    }

    class Judge:
        model = "fake-judge"

        def __init__(self):
            self.calls = 0

        def __call__(self, system, user, max_tokens=700):
            self.calls += 1
            if self.calls == 2:
                raise ProviderQuotaExceeded("private quota detail")
            payload = json.loads(user)
            return {
                "key_facts": [{"fact": fact, "verdict": "yes"} for fact in payload["key_facts"]],
                "behavior": {"verdict": "yes", "reason": "ok"},
            }

    scored = ev.evaluate_pipeline_results(QUESTIONS, results, Judge())
    assert scored["attempted_questions"] == 3 and scored["pipeline_errors"] == 0
    assert scored["graded_questions"] == 1
    assert scored["grading_errors"] == 1 and scored["ungraded_questions"] == 1
    assert scored["grading"]["status"].tolist() == ["ok", "error", "skipped"]
    assert scored["behavior"]["id"].tolist() == ["q1"]


def test_run_pipeline_batch_stops_after_provider_quota_and_marks_rest_skipped():
    from src.retry import ProviderQuotaExceeded

    class Pipeline:
        def __init__(self):
            self.questions = []

        def answer(self, question):
            self.questions.append(question)
            if len(self.questions) == 2:
                raise ProviderQuotaExceeded("private provider detail")
            return {"answer": "ok", "sources": [], "latency_ms": 1}

    frame = pd.DataFrame({"id": ["q1", "q2", "q3"], "question": ["one", "two", "three"]})
    pipeline = Pipeline()
    results = ev.run_pipeline_batch(pipeline, frame)
    assert pipeline.questions == ["one", "two"]
    assert [results[qid]["status"] for qid in frame["id"]] == ["ok", "error", "skipped"]
    assert results["q2"]["error"] == "provider quota exhausted"
    assert "private provider detail" not in str(results)


def test_run_pipeline_batch_records_a_provider_api_error_and_keeps_going():
    """The first row of the dev run raised a 5xx; the old loop skipped all 167 rows after it."""
    from google.genai.errors import ServerError

    class Pipeline:
        def __init__(self):
            self.questions = []

        def answer(self, question):
            self.questions.append(question)
            if question == "one":
                raise ServerError(503, {"error": {
                    "code": 503,
                    "message": "This model is currently experiencing high demand.",
                    "status": "UNAVAILABLE",
                }})
            return {"answer": "ok", "sources": [], "abstained": False, "latency_ms": 1}

    frame = pd.DataFrame({"id": ["q1", "q2", "q3"], "question": ["one", "two", "three"]})
    pipeline = Pipeline()
    results = ev.run_pipeline_batch(pipeline, frame)
    assert pipeline.questions == ["one", "two", "three"]
    assert [results[qid]["status"] for qid in frame["id"]] == ["error", "ok", "ok"]
    assert results["q1"]["error"] == (
        "ServerError (HTTP 503): This model is currently experiencing high demand."
    )


def test_run_pipeline_batch_records_a_non_provider_failure_without_stopping():
    class Pipeline:
        def answer(self, question):
            if question == "one":
                raise ValueError("citation validation failed")
            return {"answer": "ok", "sources": [], "abstained": False, "latency_ms": 1}

    frame = pd.DataFrame({"id": ["q1", "q2"], "question": ["one", "two"]})
    results = ev.run_pipeline_batch(Pipeline(), frame)
    assert [results[qid]["status"] for qid in frame["id"]] == ["error", "ok"]
    assert results["q1"]["error"] == "ValueError: citation validation failed"
