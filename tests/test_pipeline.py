import pytest

from src import config
from src.pipeline import (
    ABSTAIN_CITATION_VALIDATION_FAILED,
    ABSTAIN_INSUFFICIENT_EVIDENCE,
    ABSTAIN_MESSAGE,
    ABSTAIN_UNCALIBRATED,
    DISCLAIMER,
    RAGPipeline,
    format_result,
)
from tests.conftest import FakeClient


def test_abstains_without_calling_the_llm_when_evidence_is_weak(make_pipeline):
    pipeline, client = make_pipeline(threshold=99.0)   # no passage can reach this score
    result = pipeline.answer("diabetes insulin")
    assert result["abstained"] is True
    assert result["answer"] == ABSTAIN_MESSAGE
    assert result["sources"] == [] and result["passages"] == []
    assert client.calls == []   # the model is never asked to answer without evidence


def test_abstention_message_points_to_emergency_help():
    assert "emergency" in ABSTAIN_MESSAGE


def test_answers_when_the_best_score_reaches_the_threshold(make_pipeline):
    pipeline, client = make_pipeline(threshold=1.0, answer="Diabetes affects blood sugar [1].")
    result = pipeline.answer("diabetes insulin")
    assert result["abstained"] is False
    assert result["answer"] == "Diabetes affects blood sugar [1]."
    assert len(client.calls) == 1
    assert result["disclaimer"] == DISCLAIMER


def test_a_score_equal_to_the_threshold_counts_as_enough_evidence(retriever):
    best = retriever.retrieve("diabetes insulin")[0]["rerank_score"]
    pipeline = RAGPipeline(retriever, FakeClient(), threshold=best, calibrated=True)
    assert pipeline.answer("diabetes insulin")["abstained"] is False


def test_sources_are_exactly_the_passages_that_were_in_the_prompt(make_pipeline):
    pipeline, client = make_pipeline()
    result = pipeline.answer("diabetes insulin")
    prompt = client.calls[0]["contents"]

    assert len(result["sources"]) == 1  # only [1] was cited, though all five passages were in the prompt
    assert f"[{config.CONTEXT_K}]" in prompt and f"[{config.CONTEXT_K + 1}]" not in prompt
    for source, passage in zip(result["sources"], result["passages"]):
        assert source["title"] == passage["title"] and source["url"] == passage["url"]
        assert passage["text"] in prompt
    assert [s["n"] for s in result["sources"]] == [1]


def test_sources_keep_original_prompt_numbers_and_citation_order(make_pipeline):
    pipeline, _ = make_pipeline(answer="First [3], then [1].")
    result = pipeline.answer("diabetes insulin")
    assert [source["n"] for source in result["sources"]] == [3, 1]
    assert [source["title"] for source in result["sources"]] == [
        result["passages"][2]["title"], result["passages"][0]["title"]
    ]


def test_a_blank_question_is_rejected(make_pipeline):
    with pytest.raises(ValueError):
        make_pipeline()[0].answer("   ")


def test_comma_and_range_citations_are_accepted_and_listed_as_sources(make_pipeline):
    pipeline, _ = make_pipeline(answer="Fever [1, 2]. Cough [1-3].")
    result = pipeline.answer("diabetes insulin")
    assert result["abstained"] is False
    assert [source["n"] for source in result["sources"]] == [1, 2, 3]


def test_a_bracketed_year_never_becomes_a_source(make_pipeline):
    pipeline, _ = make_pipeline(answer="In 2024 this changed [2024]. Fever is common [1].")
    result = pipeline.answer("diabetes insulin")
    assert result["abstained"] is False
    assert [source["n"] for source in result["sources"]] == [1]


def test_abstain_reason_separates_the_threshold_from_the_citation_check(make_pipeline):
    weak, _ = make_pipeline(threshold=99.0)
    assert weak.answer("diabetes insulin")["abstain_reason"] == ABSTAIN_INSUFFICIENT_EVIDENCE
    uncited, _ = make_pipeline(answer="An answer with no citation at all.")
    result = uncited.answer("diabetes insulin")
    assert result["abstained"] is True and result["citation_validation_failed"] is True
    assert result["abstain_reason"] == ABSTAIN_CITATION_VALIDATION_FAILED


def test_an_uncalibrated_pipeline_reports_that_reason(retriever):
    pipeline = RAGPipeline(retriever, FakeClient(), threshold=0.0, calibrated=False)
    result = pipeline.answer("diabetes insulin")
    assert result["abstained"] is True and result["abstain_reason"] == ABSTAIN_UNCALIBRATED


def test_an_emergency_has_no_abstain_reason(make_pipeline):
    result = make_pipeline()[0].answer("I have crushing chest pain right now")
    assert result["emergency"] is True and result["abstain_reason"] is None


def test_no_retrieved_passages_means_abstain():
    class NothingFound:
        def retrieve(self, question):
            return []

    client = FakeClient()
    result = RAGPipeline(NothingFound(), client, threshold=-100.0, calibrated=True).answer("anything")
    assert result["abstained"] is True and client.calls == []


def test_format_result_lists_numbered_sources_with_urls_then_the_disclaimer(make_pipeline):
    text = format_result(make_pipeline()[0].answer("diabetes insulin"))
    assert "Sources:" in text and "1. " in text and "https://example.org/" in text
    assert text.rstrip().endswith(DISCLAIMER)


def test_an_emergency_never_reaches_retrieval_or_the_llm(make_pipeline):
    pipeline, client = make_pipeline()

    def boom(*args, **kwargs):
        raise AssertionError("retriever must not run for an emergency")

    pipeline.retriever.retrieve = boom
    result = pipeline.answer("I have crushing chest pain that spreads to my left arm")
    assert result["emergency"] is True and result["abstained"] is False and result["redirected"] is False
    assert result["sources"] == [] and result["passages"] == [] and result["top_score"] is None
    assert "emergency number" in result["answer"].lower()
    assert client.calls == []


def test_a_suicidal_message_gets_the_crisis_lines(make_pipeline):
    result = make_pipeline()[0].answer("I want to kill myself")
    assert result["emergency"] is True and "988" in result["answer"] and "16328" in result["answer"]


def test_a_harmful_request_is_refused_without_retrieval_or_llm(make_pipeline):
    pipeline, client = make_pipeline()
    pipeline.retriever.retrieve = lambda question: (_ for _ in ()).throw(AssertionError("no retrieval"))
    result = pipeline.answer("How can I make a poison that cannot be detected?")
    assert result["refused"] is True
    assert result["emergency"] is False and result["redirected"] is False and result["abstained"] is False
    assert "can't help" in result["answer"] and result["sources"] == []
    assert client.calls == []


def test_a_personal_advice_question_is_redirected_without_retrieval_or_llm(make_pipeline):
    pipeline, client = make_pipeline()
    pipeline.retriever.retrieve = lambda question: (_ for _ in ()).throw(AssertionError("no retrieval"))
    result = pipeline.answer("How much ibuprofen can I give my 2-year-old?")
    assert result["redirected"] is True and result["emergency"] is False and result["abstained"] is False
    assert "doctor or pharmacist" in result["answer"] and result["sources"] == []
    assert client.calls == []


def test_a_normal_question_is_neither_emergency_nor_redirect(make_pipeline):
    result = make_pipeline()[0].answer("What are the symptoms of type 2 diabetes?")
    assert result["emergency"] is False and result["redirected"] is False and result["refused"] is False


def test_an_educational_question_about_children_still_gets_an_answer(make_pipeline):
    pipeline, client = make_pipeline()
    result = pipeline.answer("diabetes insulin in children")
    assert result["redirected"] is False and len(client.calls) == 1


def test_threshold_is_not_trusted_until_calibration_metadata_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "THRESHOLD_PATH", tmp_path / "abstention_threshold.json")
    monkeypatch.setattr(config, "DEV_EVAL_PATH", tmp_path / "questions.csv")
    monkeypatch.setattr(config, "INDEX_MANIFEST_PATH", tmp_path / "manifest.json")
    config.DEV_EVAL_PATH.write_text("id,question\nq1,example?\n", encoding="utf-8")
    config.INDEX_MANIFEST_PATH.write_text("{}", encoding="utf-8")

    assert config.load_threshold() == (config.DEFAULT_THRESHOLD, False)
    config.THRESHOLD_PATH.write_text('{"threshold": 3.25}', encoding="utf-8")
    assert config.load_threshold() == (config.DEFAULT_THRESHOLD, False)

    malformed = {"threshold": True, **config.calibration_metadata()}
    config.THRESHOLD_PATH.write_text(__import__("json").dumps(malformed), encoding="utf-8")
    assert config.load_threshold() == (config.DEFAULT_THRESHOLD, False)

    valid = {"threshold": 3.25, **config.calibration_metadata()}
    config.THRESHOLD_PATH.write_text(__import__("json").dumps(valid), encoding="utf-8")
    assert config.load_threshold() == (3.25, True)


def test_a_runtime_library_version_change_warns_instead_of_rejecting_the_threshold(tmp_path, monkeypatch, caplog):
    """torch/transformers patch releases must not take the service down; they are reported.

    The pinned model and revision are still hard matches, so a version bump alone cannot silently
    change which model is served - it is only a hint that scores may have drifted.
    """
    monkeypatch.setattr(config, "THRESHOLD_PATH", tmp_path / "abstention_threshold.json")
    monkeypatch.setattr(config, "DEV_EVAL_PATH", tmp_path / "questions.csv")
    monkeypatch.setattr(config, "INDEX_MANIFEST_PATH", tmp_path / "manifest.json")
    config.DEV_EVAL_PATH.write_text("id,question\nq1,example?\n", encoding="utf-8")
    config.INDEX_MANIFEST_PATH.write_text("{}", encoding="utf-8")
    config.THRESHOLD_PATH.write_text(
        __import__("json").dumps({"threshold": 3.25, **config.calibration_metadata()}), encoding="utf-8"
    )
    monkeypatch.setattr(config, "model_runtime_versions", lambda: {"sentence-transformers": "changed"})
    assert config.load_threshold() == (3.25, True)
    assert "different runtime versions" in caplog.text


def test_threshold_is_invalidated_when_the_safety_rules_version_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "THRESHOLD_PATH", tmp_path / "abstention_threshold.json")
    monkeypatch.setattr(config, "DEV_EVAL_PATH", tmp_path / "questions.csv")
    monkeypatch.setattr(config, "INDEX_MANIFEST_PATH", tmp_path / "manifest.json")
    config.DEV_EVAL_PATH.write_text("id,question\nq1,example?\n", encoding="utf-8")
    config.INDEX_MANIFEST_PATH.write_text("{}", encoding="utf-8")
    config.THRESHOLD_PATH.write_text(
        __import__("json").dumps({"threshold": 3.25, **config.calibration_metadata()}), encoding="utf-8"
    )
    assert config.load_threshold() == (3.25, True)

    monkeypatch.setattr(config, "SAFETY_RULES_VERSION", "a-newer-ruleset")
    assert config.load_threshold() == (config.DEFAULT_THRESHOLD, False)


def test_the_safety_fingerprint_is_a_version_not_a_file_hash(tmp_path, monkeypatch):
    """A comment or a line-ending change must not invalidate a calibration."""
    monkeypatch.setattr(config, "THRESHOLD_PATH", tmp_path / "abstention_threshold.json")
    monkeypatch.setattr(config, "DEV_EVAL_PATH", tmp_path / "questions.csv")
    monkeypatch.setattr(config, "INDEX_MANIFEST_PATH", tmp_path / "manifest.json")
    config.DEV_EVAL_PATH.write_text("id,question\nq1,example?\n", encoding="utf-8")
    config.INDEX_MANIFEST_PATH.write_text("{}", encoding="utf-8")
    metadata = config.calibration_metadata()
    assert metadata["safety_rules_version"] == config.SAFETY_RULES_VERSION
    assert "safety_rules_sha256" not in metadata
    config.THRESHOLD_PATH.write_text(
        __import__("json").dumps({"threshold": 3.25, **metadata}), encoding="utf-8"
    )
    # A CRLF copy of the same evaluation file hashes to the same value, so a Windows checkout keeps
    # the calibration while an actual label change still invalidates it.
    config.DEV_EVAL_PATH.write_bytes(b"id,question\r\nq1,example?\r\n")
    assert config.load_threshold() == (3.25, True)
    config.DEV_EVAL_PATH.write_text("id,question\nq1,changed?\n", encoding="utf-8")
    assert config.load_threshold() == (config.DEFAULT_THRESHOLD, False)


def test_a_threshold_for_another_reranker_or_changed_eval_data_is_not_used(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "THRESHOLD_PATH", tmp_path / "abstention_threshold.json")
    monkeypatch.setattr(config, "DEV_EVAL_PATH", tmp_path / "questions.csv")
    monkeypatch.setattr(config, "INDEX_MANIFEST_PATH", tmp_path / "manifest.json")
    config.DEV_EVAL_PATH.write_text("id,question\nq1,example?\n", encoding="utf-8")
    config.INDEX_MANIFEST_PATH.write_text("{}", encoding="utf-8")

    stale = {"threshold": 3.25, **config.calibration_metadata(), "reranker_model": "some/other-reranker"}
    config.THRESHOLD_PATH.write_text(__import__("json").dumps(stale), encoding="utf-8")
    assert config.load_threshold() == (config.DEFAULT_THRESHOLD, False)

    valid = {"threshold": 3.25, **config.calibration_metadata()}
    config.THRESHOLD_PATH.write_text(__import__("json").dumps(valid), encoding="utf-8")
    assert config.load_threshold() == (3.25, True)
    config.DEV_EVAL_PATH.write_text("id,question\nq1,changed?\n", encoding="utf-8")
    assert config.load_threshold() == (config.DEFAULT_THRESHOLD, False)
