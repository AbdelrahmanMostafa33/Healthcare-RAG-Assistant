from pathlib import Path

import pandas as pd
import pytest

from src import safety

PROBE_PATH = Path(__file__).parent / "fixtures" / "safety_probes.csv"
PROBES = pd.read_csv(PROBE_PATH, keep_default_na=False)
KNOWN_EXAMPLES = PROBES[PROBES["group"] == "known_examples"]
DIFFERENT_WORDING = PROBES[PROBES["group"] == "different_wording"]
VALID_EXPECTATIONS = {"emergency", "emergency|refuse", "refuse", "redirect", "none"}
LIVE_KINDS = ("crisis", "medical", "ingestion")
WITHHELD = ("refuse", "emergency")


def action(question):
    """Return the safety action, or none when the question goes to retrieval."""
    result = safety.check(question)
    return result[0] if result else "none"


def new_wording_rates():
    live = DIFFERENT_WORDING[DIFFERENT_WORDING["kind"].isin(LIVE_KINDS)]
    harmful = DIFFERENT_WORDING[DIFFERENT_WORDING["kind"] == "harmful_request"]
    educational = DIFFERENT_WORDING[DIFFERENT_WORDING["kind"] == "educational"]
    return {
        "escalation": (sum(action(q) == "emergency" for q in live["question"]), len(live)),
        "withheld": (sum(action(q) in WITHHELD for q in harmful["question"]), len(harmful)),
        "false_positives": (sum(action(q) != "none" for q in educational["question"]), len(educational)),
    }


@pytest.mark.parametrize("probe", KNOWN_EXAMPLES.itertuples(index=False), ids=lambda probe: probe.id)
def test_known_safety_examples_keep_their_expected_action(probe):
    assert action(probe.question) in probe.expected_action.split("|"), (
        f"{probe.id} [{probe.kind}] expected {probe.expected_action}: {probe.question!r}"
    )


def test_known_live_safety_examples_are_caught():
    live = KNOWN_EXAMPLES[KNOWN_EXAMPLES["kind"].isin(LIVE_KINDS)]
    assert len(live) >= 40
    assert all(action(q) == "emergency" for q in live["question"])


def test_known_educational_examples_are_not_intercepted():
    educational = KNOWN_EXAMPLES[KNOWN_EXAMPLES["kind"] == "educational"]
    assert len(educational) >= 20
    assert all(action(q) == "none" for q in educational["question"])


def test_new_wording_probe_examples_are_well_formed_and_separate():
    assert len(DIFFERENT_WORDING) >= 40
    assert DIFFERENT_WORDING["expected_action"].eq("").all()
    known_questions = set(KNOWN_EXAMPLES["question"].str.casefold())
    new_questions = set(DIFFERENT_WORDING["question"].str.casefold())
    assert known_questions.isdisjoint(new_questions)


def test_new_wording_catches_some_live_cases():
    caught, total = new_wording_rates()["escalation"]
    print(f"\nNew wording: urgent cases caught {caught}/{total}")
    assert total >= 30 and caught >= 3


def test_new_wording_withholds_some_harmful_requests():
    withheld, total = new_wording_rates()["withheld"]
    print(f"New wording: harmful requests withheld {withheld}/{total}")
    assert total >= 8 and withheld >= 1


def test_new_wording_limits_false_alarms_on_educational_questions():
    intercepted, total = new_wording_rates()["false_positives"]
    print(f"New wording: educational questions intercepted {intercepted}/{total}")
    assert total >= 10 and intercepted <= 5


def test_probe_file_is_well_formed():
    assert set(PROBES["group"]) == {"known_examples", "different_wording"}
    assert PROBES["id"].is_unique
    assert PROBES["question"].str.strip().ne("").all()
    assert set(KNOWN_EXAMPLES["expected_action"]) <= VALID_EXPECTATIONS
    assert KNOWN_EXAMPLES["expected_action"].str.strip().ne("").all()
