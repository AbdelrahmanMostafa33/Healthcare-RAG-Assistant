"""Tests for the pre-retrieval emergency detector."""
import pytest

from src.emergency import detect_emergency, emergency_response


@pytest.mark.parametrize("query,expected", [
    # DIRECT — no personal marker needed
    ("I am suicidal and I don't know what to do", "suicide"),
    ("I want to kill myself", "suicide"),
    ("I've been self-harming and can't stop", "suicide"),
    ("I can't breathe and my chest feels tight", "breathing"),
    ("My father stopped breathing a minute ago", "breathing"),
    ("My throat is closing after eating peanuts", "anaphylaxis"),

    # CONTEXTUAL — personal marker required
    ("I have crushing chest pain that spreads to my left arm", "cardiac"),
    ("My husband has chest pain and is sweating", "cardiac"),
    ("I think I'm having a stroke", "stroke"),
    ("My mother's face is drooping and her speech is slurred", "stroke"),
    ("I have severe bleeding and it won't stop", "bleeding"),
    ("My son is having a seizure right now", "seizure"),
    ("He is unconscious and won't wake up", "unconscious"),
    ("I think I overdosed on my medication", "overdose"),
])
def test_detects_emergency(query, expected):
    assert detect_emergency(query) == expected


@pytest.mark.parametrize("query", [
    "",
    "What are the symptoms of type 2 diabetes?",
    "How is high blood pressure treated?",
    "What are the symptoms of a heart attack?",       # informational, no personal marker
    "What is a stroke and how is it treated?",        # informational
    "How can seizures be prevented?",                 # informational
    "What should I eat to lower cholesterol?",
    "How many mg of ibuprofen should I give my 4 year old?",
    "What is the treatment for an overdose?",         # informational
])
def test_does_not_trigger_on_informational(query):
    assert detect_emergency(query) is None


def test_default_response_used_for_unknown_type():
    assert "emergency number" in emergency_response("something_else")


def test_suicide_response_has_crisis_line():
    response = emergency_response("suicide")
    assert "988" in response
    assert "not alone" in response.lower()


# ---- Pipeline integration ----------------------------------------------------

def test_pipeline_short_circuits_on_emergency(make_pipeline):
    pipeline, client = make_pipeline()
    result = pipeline.answer("I have crushing chest pain that spreads to my left arm")

    assert result["emergency"] is True
    assert result["abstained"] is False
    assert result["sources"] == []
    assert result["passages"] == []
    assert result["top_score"] is None
    assert client.calls == []   # the LLM was never called
    assert "emergency number" in result["answer"].lower()


def test_pipeline_retrieval_not_called_on_emergency(make_pipeline):
    pipeline, _ = make_pipeline()

    def boom(*args, **kwargs):
        raise AssertionError("retriever must not run for an emergency")

    pipeline.retriever.retrieve = boom
    result = pipeline.answer("I am suicidal and I don't know what to do")

    assert result["emergency"] is True


def test_pipeline_normal_question_is_not_emergency(make_pipeline):
    pipeline, _ = make_pipeline()
    result = pipeline.answer("What are the symptoms of type 2 diabetes?")
    assert result["emergency"] is False


def test_pipeline_emergency_flagged_off_by_config(make_pipeline, monkeypatch):
    from src import config
    monkeypatch.setattr(config, "EMERGENCY_ENABLED", False)
    pipeline, _ = make_pipeline()
    result = pipeline.answer("I have crushing chest pain that spreads to my left arm")
    assert result["emergency"] is False   # falls through to the normal path


def test_suicide_question_returns_crisis_line(make_pipeline):
    pipeline, _ = make_pipeline()
    result = pipeline.answer("I want to kill myself")
    assert result["emergency"] is True
    assert "988" in result["answer"]