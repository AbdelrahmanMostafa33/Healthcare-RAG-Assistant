import pandas as pd
import pytest

from src import evaluation as ev


def test_choose_threshold_separates_clean_data_with_a_margin():
    best = ev.choose_threshold([5.0, 6.0, 7.0, 4.0], [-3.0, -1.0, 0.5])
    assert best["balanced_accuracy"] == 1.0
    assert 0.5 < best["threshold"] <= 4.0
    assert best["answerable_answered"] == 1.0 and best["unanswerable_abstained"] == 1.0


def test_choose_threshold_with_overlap_balances_both_errors():
    # one unanswerable question scores high, one answerable scores low
    best = ev.choose_threshold([6.0, 5.0, 4.0, -1.0], [-4.0, -3.0, -2.0, 4.5])
    assert best["balanced_accuracy"] == pytest.approx(0.875)
    assert best["n_answerable"] == 4 and best["n_unanswerable"] == 4


def test_choose_threshold_prefers_the_cautious_value_on_ties():
    # The scores do not separate the groups at all, so "answer everything" and
    # "abstain on everything" tie at 0.5. The higher (more cautious) one must win.
    best = ev.choose_threshold([0.0, 1.0], [2.0, 3.0])
    assert best["balanced_accuracy"] == 0.5
    assert best["threshold"] == 4.0


def test_choose_threshold_needs_both_groups():
    with pytest.raises(ValueError):
        ev.choose_threshold([], [1.0])


def test_abstention_metrics():
    expected = ["answer", "answer", "answer", "insufficient_evidence", "insufficient_evidence", "emergency"]
    abstained = [False, True, False, True, False, False]
    m = ev.abstention_metrics(expected, abstained)
    assert m["abstention_recall"] == pytest.approx(0.5)       # 1 of 2 unanswerable abstained
    assert m["abstention_precision"] == pytest.approx(0.5)    # 1 of 2 abstentions was right
    assert m["over_refusal"] == pytest.approx(1 / 3)          # 1 of 3 answerable refused


def test_abstention_metrics_handle_missing_denominators():
    m = ev.abstention_metrics(["answer", "answer"], [False, False])
    assert m["abstention_recall"] is None and m["abstention_precision"] is None and m["over_refusal"] == 0.0


def test_answer_claims_drop_citation_markers_and_short_fragments():
    answer = "Fever is a common symptom of the flu [1]. See a doctor.\n• Rest and drinking fluids may help [2][3] a lot.\n- ok"
    assert ev.answer_claims(answer) == [
        "Fever is a common symptom of the flu.",
        "Rest and drinking fluids may help a lot.",
    ]


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
])


def test_score_key_facts_maps_verdicts_to_scores_and_skips_questions_without_facts():
    judge = FakeJudge({ev.FACT_SYSTEM: {"results": [{"verdict": "yes"}, {"verdict": "partial"}]}})
    rows = ev.score_key_facts(judge, QUESTIONS, {"q1": "Flu is caused by a virus.", "q2": "x"})
    assert rows["id"].tolist() == ["q1", "q1"]
    assert rows["score"].tolist() == [1.0, 0.5]
    assert rows["judge_model"].unique().tolist() == ["fake-judge"]


def test_missing_judge_verdicts_count_as_not_found():
    judge = FakeJudge({ev.FACT_SYSTEM: {"results": [{"verdict": "yes"}]}})     # judge forgot the second fact
    rows = ev.score_key_facts(judge, QUESTIONS, {"q1": "a", "q2": "b"})
    assert rows["score"].tolist() == [1.0, 0.0]


def test_score_behavior_uses_the_rubric_for_the_expected_behavior():
    judge = FakeJudge({ev.BEHAVIOR_SYSTEM: {"verdict": "Yes", "reason": "abstained"}})
    rows = ev.score_behavior(judge, QUESTIONS, {"q1": "a", "q2": "I couldn't find enough information."})
    assert rows["ok"].tolist() == [True, True]
    assert rows["expected"].tolist() == ["answer", "insufficient_evidence"]


def test_score_retrieval_computes_recall_and_reciprocal_rank():
    judge = FakeJudge({
        ev.CONTEXT_FACT_SYSTEM: {"results": [{"verdict": "yes"}, {"verdict": "no"}]},
        ev.PASSAGE_SYSTEM: {"results": [{"passage": 1, "verdict": "partial"}, {"passage": 2, "verdict": "yes"},
                                        {"passage": 3, "verdict": "yes"}]},
    })
    passages = {"q1": [{"text": "a"}, {"text": "b"}, {"text": "c"}]}
    row = ev.score_retrieval(judge, QUESTIONS, passages).iloc[0]
    assert row["recall"] == pytest.approx(0.5)
    assert row["rr"] == pytest.approx(0.5)                    # first clearly-relevant passage is rank 2
    assert row["fact_scores"] == [1.0, 0.0]


def test_score_groundedness_skips_abstentions_and_counts_unanswered_claims_as_unsupported():
    judge = FakeJudge({ev.CLAIM_SYSTEM: {"results": [{"verdict": "yes"}]}})   # only 1 of 2 claims judged
    results = {
        "q1": {"abstained": False, "passages": [{"text": "evidence"}],
               "answer": "The flu is caused by a virus [1]. It spreads easily between people [1]."},
        "q2": {"abstained": True, "passages": [], "answer": "I couldn't find enough information in my sources."},
    }
    rows = ev.score_groundedness(judge, QUESTIONS, results)
    assert rows["id"].tolist() == ["q1", "q1"]
    assert rows["score"].tolist() == [1.0, 0.0]
