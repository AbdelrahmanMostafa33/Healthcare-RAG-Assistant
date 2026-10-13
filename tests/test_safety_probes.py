"""Held-out safety probes, kept separate from the rules they check.

Every row in data/eval/safety_probes.csv names the action the pre-retrieval rules must return.
Rows of kind ``educational`` must return nothing: a question about a topic is not a person in
trouble, and sending one to a crisis page is its own kind of failure.

The point of keeping these in data is that a future pattern edit has to face them. When adding a
rule, add probes for the phrasings it should and should not catch, and keep the counts documented
in data/eval/README.md honest.
"""
import pandas as pd
import pytest

from src import config, safety

PROBE_PATH = config.EVAL_PATH.parent / "safety_probes.csv"
PROBES = pd.read_csv(PROBE_PATH, keep_default_na=False)

VALID_EXPECTATIONS = {"emergency", "emergency|refuse", "refuse", "redirect", "none"}


def action(question):
    """The rule's decision, flattened to one word (``none`` means it reaches retrieval)."""
    result = safety.check(question)
    return result[0] if result else "none"


@pytest.mark.parametrize("probe", PROBES.itertuples(index=False), ids=lambda probe: probe.id)
def test_probe_returns_the_expected_action(probe):
    assert action(probe.question) in probe.expected_action.split("|"), (
        f"{probe.id} [{probe.kind}] expected {probe.expected_action}: {probe.question!r}"
    )


def test_every_live_emergency_probe_is_caught():
    """A missed crisis is the most expensive mistake this module can make."""
    live = PROBES[PROBES["kind"].isin(["crisis", "medical", "ingestion"])]
    assert len(live) >= 40
    missed = [row.question for row in live.itertuples() if action(row.question) != "emergency"]
    assert missed == []


def test_no_educational_probe_is_intercepted():
    educational = PROBES[PROBES["kind"] == "educational"]
    assert len(educational) >= 20
    intercepted = [row.question for row in educational.itertuples() if action(row.question) != "none"]
    assert intercepted == []


def test_no_harmful_request_probe_is_answered():
    """Refusing and redirecting to a crisis line both withhold the method; answering does not."""
    harmful = PROBES[PROBES["kind"] == "harmful_request"]
    answered = [
        row.question for row in harmful.itertuples()
        if action(row.question) not in row.expected_action.split("|")
    ]
    assert answered == []


def test_the_probe_file_is_well_formed():
    assert len(PROBES) >= 60
    assert PROBES["id"].is_unique
    assert set(PROBES["expected_action"]) <= VALID_EXPECTATIONS
    assert PROBES["question"].str.strip().ne("").all()
