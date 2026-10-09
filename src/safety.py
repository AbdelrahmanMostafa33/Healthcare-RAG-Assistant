"""Rule-based checks that run before retrieval.

Two checks, in this order:

1. Emergency: the person describes an acute emergency happening now (chest pain, stroke signs,
   an overdose, thoughts of suicide). The answer is a fixed message with emergency numbers.
2. Personal advice: the question asks for a dose, a diagnosis or a treatment decision about one
   specific person. A static knowledge base cannot answer that, so the answer is a fixed
   redirect to a doctor or pharmacist.

Both are deliberately narrow keyword rules, not a classifier. A question that matches nothing goes
through retrieval, where the evidence threshold and the prompt rules still apply. A wrong match is
the costly mistake (an educational question gets a fixed reply), so the emergency phrases only count
when the question is written about a real person ("I", "my", "he"), and every rule is tested
against educational questions it must not catch (tests/test_safety.py).
"""
import re

# The question describes a real person's situation, not a topic. A bare "I" is not enough:
# "What should I know about chest pain?" is an educational question.
PERSONAL = re.compile(
    r"\b(i have|i've|ive|i am|i'm|im|i feel|i think|i just|i was|i keep|i took|i can't|i cant|i cannot"
    r"|my|myself|he|she|he's|she's|his|her)\b"
)

# Phrases that already say "me", so no separate personal marker is needed.
CRISIS_ALWAYS = [
    "kill myself", "end my life", "take my own life", "want to die", "wanna die",
    "hurt myself", "harm myself", "cut myself", "better off dead",
]
CRISIS_WITH_MARKER = ["suicidal", "suicide", "self-harm", "self harm"]

# Acute-emergency descriptions. They only count together with a personal marker, so
# "What are the symptoms of a heart attack?" or "What is anaphylaxis?" are answered normally.
MEDICAL_PHRASES = [
    "can't breathe", "cant breathe", "cannot breathe", "can not breathe", "unable to breathe",
    "struggling to breathe", "gasping", "stopped breathing", "not breathing",
    "anaphylaxis", "anaphylactic", "throat is closing", "throat closing", "throat is swelling",
    "took too many", "took too much", "overdosed", "poisoned",
    "bottle of pills", "handful of pills", "handful of tablets",
    "chest pain", "chest pressure", "chest tightness", "chest heaviness", "crushing chest",
    "chest hurts", "chest is tight", "pain in my chest", "pressure in my chest", "can't feel my arm", "can't feel my left arm",
    "having a heart attack", "having a stroke", "face drooping", "face is drooping", "face is numb",
    "slurred speech", "can't speak", "sudden weakness on one side",
    "severe bleeding", "heavy bleeding", "won't stop bleeding", "wont stop bleeding",
    "bleeding heavily", "bleeding a lot",
    "having a seizure", "having a fit", "convulsion",
    "unconscious", "unresponsive", "passed out", "won't wake", "wont wake", "no pulse",
    "worst headache", "thunderclap",
]
MEDICAL_PATTERNS = [
    # swallowed something dangerous
    re.compile(r"swallowed\b.*\b(pills?|tablets?|capsules?|medicine|medication|batter(y|ies)|bleach|poison|chemicals?|detergent)\b"),
    # a baby with a fever
    re.compile(r"\b(newborn|infant|baby|\d+[- ]?(day|week)s?[- ]?old|[1-3][- ]?months?[- ]?old)\b.*\b(fever|temperature)\b"),
    re.compile(r"\b(fever|temperature)\b.*\b(newborn|infant|baby|\d+[- ]?(day|week)s?[- ]?old|[1-3][- ]?months?[- ]?old)\b"),
    # a diabetic who is very unwell
    re.compile(r"\b(diabetic|blood sugar|glucose)\b.{0,80}\b(vomiting|very drowsy|confused|unconscious)\b"),
]

EMERGENCY_RESPONSES = {
    "crisis": (
        "**You are not alone, and help is available right now.**\n\n"
        "Please reach out to a crisis line immediately:\n"
        "- Egypt: **08008880700** (Ministry of Health)\n"
        "- US: **988** (Suicide & Crisis Lifeline)\n"
        "- UK: **116 123** (Samaritans)\n"
        "- International: https://findahelpline.com\n\n"
        "If you are in immediate danger, call your local emergency number "
        "(Egypt **123**, US **911**, EU **112**, UK **999**)."
    ),
    "medical": (
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

REDIRECT_MESSAGE = (
    "This looks like a question about your own situation or a specific person. This "
    "assistant gives general health information only — it cannot look at your symptoms, "
    "results or circumstances, and it does not give personal medical advice or doses. Please "
    "speak to a doctor or pharmacist about your own situation. If you think this could "
    "be an emergency, call your local emergency number or go to the nearest emergency "
    "department now."
)

# Personal-advice patterns. Each one needs a first-person or "my child" wording; a bare topic
# ("children", "insulin", "is it safe to exercise") never matches on its own.
_CHILD = r"(\d+[- ]?(year|yr|month|week)s?[- ]?old|child|son|daughter|baby|toddler|kid)"
_DECISION_VERBS = r"(take|give|stop|skip|double|mix|combine|keep taking|continue taking)"
PERSONAL_ADVICE_PATTERNS = [re.compile(p) for p in [
    # "Can I stop taking insulin?", "Should I take a double dose?"
    rf"\b(should|can|could|may) i {_DECISION_VERBS}\b",
    rf"\bis it (ok|okay|safe|fine) (for me )?to {_DECISION_VERBS}\b",
    # amounts for a person: "how much can I give my son", "what dose should I take", "5 mg"
    r"\bhow (much|many)\b[^?.!]*\b(should|can|do) i (give|take)\b",
    r"\bwhat (dose|dosage|amount)\b[^?.!]*\b(should|can) i\b",
    r"\b\d+(\.\d+)?\s*(mg|mcg|µg|μg|ml)\b",
    # swapping or skipping a prescribed treatment
    r"\b(instead of|in place of|rather than|replace|replacing|substitute)\b[^?.!]*\b(medication|medicine|prescription|insulin|treatment|drug)s?\b",
    r"\bshould i (have|get|undergo) (the |an? )?(surgery|operation|biopsy)\b",
    r"\b(which|what) (treatment|medicine|medication|drug|therapy) (is|would be) (best|right|better) for my\b",
    # a child with a current problem: "my 3 year old has a rash"
    rf"\bmy {_CHILD}\b[^?.!]*\b(has|have|had|keeps|cannot|can't|won't)\b",
    # own results with a number, or "my lab results"
    r"\bmy (hba1c|a1c|egfr|cholesterol|blood pressure|blood sugar|glucose|creatinine|tsh|psa)\b[^?]*\d",
    r"\b(my|these|this) (lab|test|blood test|scan|x-?ray|mri|ct|biopsy) (results?|report)\b",
    # asking whether the person has a condition
    r"\b(do|could|might) i have\b(?! to\b)",
    r"\b(does|could|might) (he|she) have\b",
    r"\bwhat (disease|condition|illness|infection|disorder|ailment) (do )?i have\b",
    r"\bam i (having|dying|pregnant|sick)\b",
]]


def _normalise(question):
    return question.lower().replace("’", "'")


def detect_emergency(question):
    """Return "crisis", "medical" or None."""
    q = _normalise(question)
    if any(phrase in q for phrase in CRISIS_ALWAYS):
        return "crisis"
    if not PERSONAL.search(q):
        return None
    if any(phrase in q for phrase in CRISIS_WITH_MARKER):
        return "crisis"
    if any(phrase in q for phrase in MEDICAL_PHRASES) or any(p.search(q) for p in MEDICAL_PATTERNS):
        return "medical"
    return None


def is_personal_advice(question):
    q = _normalise(question)
    return any(p.search(q) for p in PERSONAL_ADVICE_PATTERNS)


def check(question):
    """Return ("emergency", message), ("redirect", message), or None if the question may go to retrieval."""
    kind = detect_emergency(question)
    if kind:
        return "emergency", EMERGENCY_RESPONSES[kind]
    if is_personal_advice(question):
        return "redirect", REDIRECT_MESSAGE
    return None
