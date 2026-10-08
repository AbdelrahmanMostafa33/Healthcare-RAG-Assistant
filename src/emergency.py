"""Emergency detection that runs before retrieval.

This is a safety net, not a triage system. It looks for a small set of phrases
that clearly describe an acute emergency and returns a fixed response instead
of going through the RAG pipeline.

Two kinds of patterns:
    DIRECT      Phrases that are almost never informational ("I am suicidal",
                "can't breathe"). Triggers on their own.
    CONTEXTUAL  Phrases that could also appear in an informational question
                ("chest pain", "having a stroke"). Triggers only when a
                personal marker ("I", "my", "he", ...) is nearby, so
                "what are the symptoms of a heart attack?" is not flagged.
"""
import re

PERSONAL_MARKERS = re.compile(
    r"\b(i|i'?m|im|i am|my|myself|me|we|our|us|he|she|they|him|her|them)\b"
)

DIRECT = {
    "suicide": [
        "suicidal", "kill myself", "end my life", "want to die",
        "self-harm", "self harm", "hurt myself", "take my own life",
    ],
    "breathing": [
        "can't breathe", "cannot breathe", "can not breathe", "unable to breathe",
        "struggling to breathe", "gasping for air", "gasping for breath",
        "stopped breathing", "not breathing",
    ],
    "anaphylaxis": [
        "anaphylaxis", "anaphylactic", "throat is closing", "throat closing",
        "severe allergic reaction",
    ],
    "overdose": [
        "took too many", "took too much", "overdosed",
        "poisoned", "swallowed",
    ],
}

CONTEXTUAL = {
    "cardiac": [
        "chest pain", "chest pressure", "chest tightness", "chest heaviness",
        "crushing chest", "having a heart attack",
    ],
    "stroke": [
        "having a stroke", "face drooping", "face is drooping", "face is numb",
        "slurred speech", "can't speak", "sudden weakness on one side",
    ],
    "bleeding": [
        "severe bleeding", "heavy bleeding", "won't stop bleeding",
        "wont stop bleeding", "bleeding heavily", "bleeding a lot",
    ],
    "seizure": [
        "having a seizure", "having a fit", "convulsion",
    ],
    "unconscious": [
        "unconscious", "unresponsive", "passed out", "won't wake",
        "wont wake", "no pulse",
    ],
    "high_fever_baby": [
        "month-old", "newborn", "infant", "baby",
    ],
    "thunderclap_headache": [
        "worst headache", "sudden",
    ],
    "diabetic_emergency": [
        "blood sugar", "drowsy",
    ],
}

RESPONSES = {
    "suicide": (
        "**You are not alone, and help is available right now.**\n\n"
        "Please reach out to a crisis line immediately:\n"
        "- Egypt: **08008880700** (Ministry of Health)\n"
        "- US: **988** (Suicide & Crisis Lifeline)\n"
        "- UK: **116 123** (Samaritans)\n"
        "- International: https://findahelpline.com\n\n"
        "If you are in immediate danger, call your local emergency number "
        "(Egypt **123**, US **911**, EU **112**, UK **999**)."
    ),
    "_default": (
        "**This may describe a medical emergency.**\n\n"
        "Please call your local emergency number immediately:\n"
        "- Egypt: **123**\n"
        "- US / Canada: **911**\n"
        "- EU: **112**\n"
        "- UK: **999**\n\n"
        "If you are with someone, stay with them. Do not drive yourself.\n\n"
        "This assistant provides general health information only and cannot "
        "help in an emergency."
    ),
}


def detect_emergency(query: str) -> str | None:
    """Return the emergency type, or None if the query is not an emergency."""
    if not query:
        return None
    q = query.lower()

    for kind, phrases in DIRECT.items():
        if any(phrase in q for phrase in phrases):
            return kind

    if not PERSONAL_MARKERS.search(q):
        return None

    for kind, phrases in CONTEXTUAL.items():
        if any(phrase in q for phrase in phrases):
            return kind

    return None


def emergency_response(kind: str) -> str:
    """Return the fixed response for an emergency type (falls back to default)."""
    return RESPONSES.get(kind, RESPONSES["_default"])