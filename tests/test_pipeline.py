import pytest

from src import config
from src.pipeline import ABSTAIN_MESSAGE, DISCLAIMER, RAGPipeline, format_result
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
    prompt = client.calls[0]["messages"][1]["content"]

    assert len(result["sources"]) == config.CONTEXT_K   # 5 sources, not the 20 FAISS candidates
    assert f"[{config.CONTEXT_K}]" in prompt and f"[{config.CONTEXT_K + 1}]" not in prompt
    for source, passage in zip(result["sources"], result["passages"]):
        assert source["title"] == passage["title"] and source["url"] == passage["url"]
        assert passage["text"] in prompt
    assert [s["n"] for s in result["sources"]] == list(range(1, config.CONTEXT_K + 1))


def test_a_blank_question_is_rejected(make_pipeline):
    with pytest.raises(ValueError):
        make_pipeline()[0].answer("   ")


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
    assert result["emergency"] is True and "988" in result["answer"] and "08008880700" in result["answer"]


def test_a_personal_advice_question_is_redirected_without_retrieval_or_llm(make_pipeline):
    pipeline, client = make_pipeline()
    pipeline.retriever.retrieve = lambda question: (_ for _ in ()).throw(AssertionError("no retrieval"))
    result = pipeline.answer("How much ibuprofen can I give my 2-year-old?")
    assert result["redirected"] is True and result["emergency"] is False and result["abstained"] is False
    assert "doctor or pharmacist" in result["answer"] and result["sources"] == []
    assert client.calls == []


def test_a_normal_question_is_neither_emergency_nor_redirect(make_pipeline):
    result = make_pipeline()[0].answer("What are the symptoms of type 2 diabetes?")
    assert result["emergency"] is False and result["redirected"] is False


def test_an_educational_question_about_children_still_gets_an_answer(make_pipeline):
    pipeline, client = make_pipeline()
    result = pipeline.answer("diabetes insulin in children")
    assert result["redirected"] is False and len(client.calls) == 1


def test_threshold_is_a_placeholder_until_notebook_04_writes_the_calibrated_one(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "THRESHOLD_PATH", tmp_path / "abstention_threshold.json")
    assert config.load_threshold() == (config.DEFAULT_THRESHOLD, False)
    (tmp_path / "abstention_threshold.json").write_text('{"threshold": 3.25}')
    assert config.load_threshold() == (3.25, True)


def test_a_threshold_calibrated_for_another_reranker_is_not_used(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "THRESHOLD_PATH", tmp_path / "abstention_threshold.json")
    (tmp_path / "abstention_threshold.json").write_text('{"threshold": 3.25, "reranker_model": "some/other-reranker"}')
    assert config.load_threshold() == (config.DEFAULT_THRESHOLD, False)
    (tmp_path / "abstention_threshold.json").write_text(f'{{"threshold": 3.25, "reranker_model": "{config.RERANKER_MODEL}"}}')
    assert config.load_threshold() == (3.25, True)
