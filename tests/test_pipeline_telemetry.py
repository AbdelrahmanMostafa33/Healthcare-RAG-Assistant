"""Pipeline telemetry: separate timings and the passages behind every decision.

An abstention used to discard the passages it had just retrieved, which made the exact context
impossible to save. These tests keep the citation-facing ``passages`` field unchanged while
requiring the retrieved evidence to be recorded.
"""


def test_an_abstention_keeps_the_passages_it_retrieved_but_cites_none(make_pipeline):
    pipeline, client = make_pipeline(threshold=99.0)
    result = pipeline.answer("diabetes insulin")

    assert result["passages"] == []          # unchanged: no answer was produced, so nothing is cited
    assert result["retrieved_passages"]      # but the evidence behind the decision is recorded
    assert result["passage_count"] == len(result["retrieved_passages"])
    assert result["retrieval_ms"] is not None
    assert result["generation_ms"] is None
    assert client.calls == []


def test_generation_and_retrieval_are_timed_separately(make_pipeline):
    result = make_pipeline()[0].answer("diabetes insulin")

    assert result["retrieval_ms"] is not None and result["generation_ms"] is not None
    assert result["passage_count"] == len(result["retrieved_passages"]) > 0
    assert result["latency_ms"] >= result["generation_ms"]
    assert result["latency_ms"] >= result["retrieval_ms"]


def test_a_citation_rejection_records_both_timings_and_the_context(make_pipeline):
    pipeline, _ = make_pipeline(answer="An answer with no citation at all.")
    result = pipeline.answer("diabetes insulin")

    assert result["citation_validation_failed"] is True
    assert result["retrieval_ms"] is not None and result["generation_ms"] is not None
    assert result["retrieved_passages"] and result["passages"] == []


def test_a_safety_reply_records_no_retrieval_or_generation_time(make_pipeline):
    result = make_pipeline()[0].answer("I want to kill myself")

    assert result["retrieved_passages"] == [] and result["passage_count"] == 0
    assert result["retrieval_ms"] is None and result["generation_ms"] is None
