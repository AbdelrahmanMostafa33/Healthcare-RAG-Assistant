import pytest

from src import config
from src.generator import SYSTEM_PROMPT, build_messages, generate, make_client
from tests.conftest import FakeClient

PASSAGES = [
    {"title": "Type 2 Diabetes", "source": "MedlinePlus", "text": "What is Type 2 Diabetes?\nIt affects how your body uses sugar."},
    {"title": "Flu", "source": "MedQuAD (CDC)", "text": "What is the flu?\nA contagious respiratory illness."},
]


def test_prompt_numbers_the_passages_and_contains_their_text():
    user = build_messages("What is diabetes?", PASSAGES)[1]["content"]
    assert "[1] Type 2 Diabetes (MedlinePlus)" in user and "It affects how your body uses sugar." in user
    assert "[2] Flu (MedQuAD (CDC))" in user
    assert user.index("[1]") < user.index("[2]") < user.index("Question: What is diabetes?")


def test_system_prompt_requires_grounding_citations_and_scope_limits():
    assert "ONLY the numbered evidence passages" in SYSTEM_PROMPT
    assert "Do not use outside medical knowledge" in SYSTEM_PROMPT
    assert "Cite the passages" in SYSTEM_PROMPT
    assert "do not give medical advice" in SYSTEM_PROMPT
    assert "no medicine or dose for a specific person" in SYSTEM_PROMPT
    assert "emergency" in SYSTEM_PROMPT
    assert "general knowledge" not in SYSTEM_PROMPT.lower()


def test_generate_makes_one_deterministic_call_with_the_configured_model():
    client = FakeClient("  Diabetes affects how the body uses sugar [1].  ")
    assert generate(client, "What is diabetes?", PASSAGES) == "Diabetes affects how the body uses sugar [1]."
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == config.LLM_MODEL and call["temperature"] == 0.0 and call["max_completion_tokens"] == config.MAX_TOKENS


def test_citations_are_left_in_the_answer():
    assert "[1]" in generate(FakeClient("Fact [1]."), "q?", PASSAGES)


@pytest.mark.parametrize("content", ["", "   ", None])
def test_an_empty_model_answer_raises_instead_of_returning_blank(content):
    with pytest.raises(RuntimeError, match="empty answer"):
        generate(FakeClient(content), "q?", PASSAGES)


def test_the_tests_do_not_need_an_api_key_to_call_generate():
    # a fake client is enough; generate() must not look at the environment
    assert generate(FakeClient("Fact [1]."), "q?", PASSAGES) == "Fact [1]."


def test_make_client_fails_fast_without_a_key():
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        make_client()


def test_make_client_points_at_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "  test-key  ")
    client = make_client()
    assert client.api_key == "test-key" and str(client.base_url).startswith(config.LLM_BASE_URL)


def test_make_client_uses_only_the_first_key_of_a_comma_separated_list(monkeypatch):
    # the old format was a comma-separated key list; sending the whole value was a 401
    monkeypatch.setenv("GROQ_API_KEY", "  gsk_first , gsk_second ,gsk_third  ")
    assert make_client().api_key == "gsk_first"


def test_make_client_rejects_a_key_of_only_commas(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", " , , ")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        make_client()
