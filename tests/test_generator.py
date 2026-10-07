import pytest

from src import config
from src.generator import SYSTEM_PROMPT, build_messages, generate
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
