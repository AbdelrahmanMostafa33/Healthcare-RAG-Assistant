"""Rule-based checks that run before retrieval.

Three checks run after normalization (typographic apostrophes are lowered to ASCII, so phone-typed text is handled). Checks run in this order:

1. Emergency: the person describes an acute emergency happening now (chest pain, stroke signs,
   an overdose, thoughts of suicide). The answer is a fixed message with emergency numbers.
2. Explicit harmful requests: requests for serious self-harm, poisoning, drug diversion, or
   prescription fraud receive a fixed refusal.
3. Personal advice: the question asks for a dose, a diagnosis or a treatment decision about one
   specific person. A static knowledge base cannot answer that, so the answer is a fixed
   redirect to a doctor or pharmacist.

These are deliberately narrow keyword rules, not a classifier. They handle curly and straight apostrophes identically, so phone-typed text reaches the same result as typed text. Generation and the API run the same normalized safety check as the evaluation harness and the notebooks, and the `src/safety.py` file on disk is the single shared module every one of them imports. A question that matches nothing goes
through retrieval, where the evidence threshold and the prompt rules still apply. A wrong match is
costly (an educational question gets a fixed reply), so the rules are split in two tiers:

* Tier 1 names a specific act or desire of the speaker in any inflection - "killing myself",
  "swallowed a bottle of pills", "I took 20 paracetamol tablets" - and needs no other context. A
  missed crisis is the worst thing this module can do, so these stand alone.
* Tier 2 matches words that also appear in lessons, poems and news ("suicide", "chest pain",
  "choking") and only counts when the question is about a real person rather than a topic.

The patterns are intentionally stem-tolerant, because the earlier phrase list missed common
inflections such as "I've been thinking about killing myself" and phone-typed apostrophes. Recall and false positives are both
measured against a held-out probe set (data/eval/safety_probes.csv, tests/test_safety_probes.py);
the notebook evaluation harness and the FastAPI service import the same `src.safety` module, so a safety fix in this file is reflected in the API, the notebooks, and the evaluation harness at once. `SAFETY_RULES_VERSION` is the calibration fingerprint for these rules: bump it when a pattern or constant changes, and a saved threshold whose version no longer matches is rejected so a deployment never runs on a threshold computed against different rules than the live code. A comment or a line-ending change does not move that version.
 """
import re

# Bump this whenever a pattern, constant or normalization in this file changes. It is what the
# calibrated threshold is keyed to. A hash of the file would also change for a typo in a comment or
# for a CRLF checkout, silently invalidating a saved calibration for a change that cannot alter a
# single safety decision.
SAFETY_RULES_VERSION = "2026-10-13.1"

# The question describes a real person's situation, not a topic. A bare "I" is not enough:
# "What should I know about chest pain?" is an educational question.
PERSONAL = re.compile(
    r"\b(i have|i've|ive|i am|i'm|im|i feel|i think|i just|i was|i keep|i took|i can't|i cant|i cannot"
    r"|my|myself|he|she|he's|she's|his|her)\b"
)

# A bare "I" is enough for the rules below, because each one already names a specific act ("swallowed
# a bottle of", "kill myself"). It is not enough for a topic word such as "chest pain", which is why
# the acute-symptom tier keeps using the stricter PERSONAL marker above: "When should I call 911 for
# chest pain?" is a question about when to call, not a person in trouble.
SPEAKER = re.compile(r"\b(i|i'm|im|i've|ive|i'd|i'll|we|we're|my|our|myself|ourselves)\b")

# Direct statements about the speaker's own life or body. These need no extra marker: they name a
# specific act or desire, and a missed crisis costs far more than an unnecessary crisis page.
CRISIS_ALWAYS = [re.compile(p) for p in [
    r"\b(kill|killing)\s+myself\b",
    # "I cut myself shaving" is a minor injury, not a crisis, so a following "shaving" is excluded.
    r"\b(cut|cutting)\s+myself\b(?!\s+(shaving|while\s+shaving))",
    r"\b(hurt|hurting|harm|harming)\s+myself\b",
    r"\b(end|ending|take|taking)\s+(my|mine)\s+(own\s+)?life\b",
    r"\b(better\s+off\s+dead|wish\s+i\s+(was|were|am)\s+dead)\b",
    r"\b(don'?t|do\s+not|dont)\s+want\s+to\s+live\b",
    r"\b(no|any)\s+(reason|point|sense)\s+(to|in)\s+((liv|continu)\w*|go(ing)?\s+on|being\s+here)\b",
    r"\b(can'?t|cannot|cant)\s+go\s+on\b",
]]

# Words that also appear in lessons, poems and news, so they only mean a crisis when the question is
# about a real person ("my friend says he wants to die").
CRISIS_WITH_MARKER = [re.compile(p) for p in [
    r"\bsuicid\w*",
    r"\bself[-\s]?harm\w*",
    r"\b(want|wants|wanna|wish|wishes|thinking|think|thought|thoughts|plan|planning)\b[^?.!]{0,30}"
    r"\b(die|dying|dead|death|suicid\w*|kill\w*|end(ing)?\s+it(\s+all)?|ending\s+(my|his|her|their)\s+life)\b",
]]
CRISIS_DESIRE_TO_DIE = re.compile(r"\b(want|wants|wanna|wish|wishes)\s+to\s+die\b")
FIRST_PERSON_DESIRE_TO_DIE = re.compile(r"\bi\b.{0,30}\b(want|wanna|wish)\b.{0,20}\bdie\b")

# Swallowing or taking far too much of something. The speaker marker is required so that "what
# happens if a child swallowed a battery?" stays a question and does not become an emergency page.
INGESTION_PATTERNS = [re.compile(p) for p in [
    r"\b(took|taken|swallow(ed|ing)?|drank|drunk|ate|eaten)\b[^?.!]{0,40}"
    r"\b(too\s+many|too\s+much|all\s+(of\s+)?(my|the)|a\s+handful|an?\s+(whole|entire)\s+bottle"
    r"|bottle\s+of|handful\s+of|packet\s+of|pack\s+of)\b[^?.!]{0,25}"
    r"\b(pills?|tablets?|capsules?|medicine|medication|paracetamol|acetaminophen|ibuprofen|aspirin"
    r"|painkillers?|opioids?|antidepressants?|sleeping\s+(pills?|tablets?)|drugs?"
    r"|bleach|detergent|chemicals?|batter(y|ies))\b",
    # A count of ten or more at once is not a dose, so "I took 20 paracetamol tablets" fires while
    # "I took 2 tablets" does not. A unit between the number and the noun is excluded, so
    # "I took 10 mg tablets" is left alone.
    r"\b(took|taken|swallow(ed|ing)?)\b[^?.!]{0,30}\b(1\d|[2-9]\d|\d{3,})\s*"
    r"(?!(mg|mcg|g|ml|units?|drops?)\b)[^?.!]{0,20}"
    r"\b(pills?|tablets?|capsules?|paracetamol|acetaminophen|ibuprofen|aspirin|painkillers?|opioids?)\b",
    # An overdose is an acute medical emergency, not a crisis call.
    r"\boverdos(e|ed|ing)\b",
]]

# Block explicit requests for actionable harm, drug diversion, or prescription fraud before
# retrieval or generation. These narrow rules target intent, not broad educational topic words.
HARMFUL_REQUEST_PATTERNS = [re.compile(p) for p in [
    r"\bhow\s+(many|much)\b.{0,80}\b(pills?|tablets?|capsules?|medicine|medication|dose|dosage"
    r"|painkillers?|paracetamol|acetaminophen|ibuprofen|aspirin|insulin|opioids?|antidepressants?"
    r"|sleeping\s+pills?|drugs?)\b.{0,60}\b(kill|die|fatal|lethal|overdose|death)\b",
    # "what is the best way to overdose" / "...to end my life"
    r"\b(what|which|where)\b[^?.!]{0,40}\b(best|quickest|easiest|fastest|painless|pain-free|reliable|surest)\b"
    r"[^?.!]{0,40}\b(way|method|technique)\b[^?.!]{0,40}\b(overdose|kill\s+(myself|himself|herself|someone)"
    r"|die|suicide|end\s+(my|his|her)\s+life)\b",
    # harming another person, but only with explicit intent to escape consequences
    r"\b(hurt|harm|poison|kill|attack)\b[^?.!]{0,40}\b(someone|somebody|another\s+person|him|her|them"
    r"|my\s+(husband|wife|partner|ex|boss|neighbour|neighbor|roommate))\b[^?.!]{0,40}"
    r"\b(get\s+away\s+with|without\s+(getting\s+caught|being\s+caught|anyone\s+knowing)|not\s+get\s+caught)\b",
    # pressuring a prescriber, gated on the deception or escalation the request is actually after
    r"\b(get|convince|persuade|pressure|trick|manipulate|mislead|lie\s+to)\b[^?.!]{0,40}"
    r"\b(my\s+)?(doctor|prescriber|psychiatrist|physician|gp)\b[^?.!]{0,60}\b(prescribe|prescribing|give\s+me)\b"
    r"[^?.!]{0,40}\b(more|extra|stronger|higher|opioids?|painkillers?|controlled|without|drugs?)\b",
    r"\b(make|create|prepare|produce|synthesize|formulate)\b.{0,80}\b(poison|toxin)\b"
    r".{0,80}\b(undetectable|cannot be detected|can't be detected|without detection)\b",
    r"\b(get|obtain|buy|acquire)\b.{0,60}\b(strong\s+)?(opioids?|controlled drugs?|prescription painkillers?)\b"
    r".{0,60}\bwithout\b.{0,30}\b(prescription|doctor|prescriber)\b",
    r"\b(fake|falsify|invent)\b.{0,60}\b(symptoms?|illness|condition)\b.{0,100}\b"
    r"(get|obtain|receive)\b.{0,60}\b(prescription|opioids?|painkillers?|controlled drugs?)\b",
]]

# Acute-emergency descriptions. They only count together with a personal marker, so
# "What are the symptoms of a heart attack?" or "What is anaphylaxis?" are answered normally.
MEDICAL_PHRASES = [
    "can't breathe", "cant breathe", "cannot breathe", "can not breathe", "unable to breathe",
    "struggling to breathe", "gasping", "stopped breathing", "not breathing",
    "isn't breathing", "isnt breathing", "is not breathing", "stopped breathing", "stops breathing",
    "choking", "collapsed", "was unresponsive",
    "anaphylaxis", "anaphylactic", "throat is closing", "throat closing", "throat is swelling",
    "severe allergic reaction", "allergic reaction",
    "took too many", "took too much", "overdosed", "poisoned",
    "bottle of pills", "handful of pills", "handful of tablets",
    "chest pain", "chest pressure", "chest tightness", "chest heaviness", "crushing chest",
    "chest hurts", "chest is tight", "pain in my chest", "pressure in my chest", "can't feel my arm", "can't feel my left arm",
    "having a heart attack", "having a stroke", "face drooping", "face is drooping", "face is numb",
    "slurred speech", "can't speak", "sudden weakness on one side",
    "severe bleeding", "heavy bleeding", "won't stop bleeding", "wont stop bleeding",
    "bleeding heavily", "bleeding a lot", "bleeding and it won't stop",
    "coughing up blood", "vomiting blood",
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
    # a road traffic collision described as happening to the speaker or their family
    re.compile(r"\b(car|motorcycle|motorbike|road|bike|cycling)\s+(accident|crash|collision)\b"),
    # breathing described as absent or failing, in any common wording
    re.compile(r"\b(isn't|isnt|is\s+not|wasn't|wasnt|not|stopped|stops|hasn't|hasnt)\b\s+breathing\b"),
    re.compile(r"\b(can't|cant|cannot|unable\s+to)\s+(breathe|breath)\b"),
]

EMERGENCY_RESPONSES = {
    "crisis": (
        "**You are not alone, and help is available right now.**\n\n"
        "Please reach out to a crisis line immediately:\n"
        "- Egypt: **16328** (Ministry of Health mental-health support)\n"
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

REFUSAL_MESSAGE = (
    "I can't help with instructions for self-harm, poisoning, misusing controlled medicines, or "
    "deceiving a clinician to obtain drugs. If anyone may be in immediate danger, call your local "
    "emergency number now. If this is about you, move away from medicines or poisons and contact "
    "someone you trust or a crisis service. Find local support at https://findahelpline.com."
)

REDIRECT_MESSAGE = (
    "This looks like a question about your own situation or a specific person. This "
    "assistant gives general health information only — it cannot look at your symptoms, "
    "results or circumstances, and it does not give personal medical advice or doses. Please "
    "speak to a doctor or pharmacist about your own situation. If you think this could "
    "be an emergency, call your local emergency number or go to the nearest emergency "
    "department now."
)

# Personal-advice patterns. Each one needs personal/specific wording; broad topics such as
# "children", "insulin" or educational explanations of units must not be redirected.
_CHILD = r"(\d+[- ]?(year|yr|month|week)s?[- ]?old|child|son|daughter|baby|toddler|kid)"
_DECISION_VERBS = r"(take|give|stop|skip|double|mix|combine|keep taking|continue taking)"
PERSONAL_ADVICE_PATTERNS = [re.compile(p) for p in [
    # "Can I stop taking insulin?", "Should I take a double dose?"
    rf"\b(should|can|could|may) i {_DECISION_VERBS}\b",
    rf"\bis it (ok|okay|safe|fine) (for me )?to {_DECISION_VERBS}\b",
    # amounts for a person: "how much can I give my son", "what dose should I take", "5 mg"
    r"\bhow (much|many)\b[^?.!]*\b(should|can|do) i (give|take)\b",
    r"\bwhat (dose|dosage|amount)\b[^?.!]*\b(should|can) i\b",
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
    # An explicitly extreme weight-loss plan is a personal health decision, not general education.
    r"\b(give|make|create|write) me (a )?plan\b.{0,80}\b(lose|weight loss)\b.{0,60}"
    r"\b(week|days?)\b.{0,80}\b(eating almost nothing|eat almost nothing|starv(e|ing))\b",
]]


def _normalise(question):
    return question.lower().replace("’", "'")


def detect_emergency(question):
    """Return "crisis", "medical" or None.

    Rules run in two tiers. Tier 1 names a specific act by the speaker ("kill myself", "swallowed a
    bottle of pills") and stands on its own. Tier 2 matches words that also appear in lessons, poems
    and news ("suicide", "chest pain") and only counts when the question is about a real person.

    Explicit educational context should not be mistaken for a current emergency solely because the
    question mentions a condition, but it never excuses a live one: "I am researching heart attacks
    for class, but I have crushing chest pain right now" is still an emergency.
    """
    q = _normalise(question)

    # Tier 1: a named act or desire of the speaker. A missed crisis is far worse than a spare one,
    # so these short-circuit everything, including explicit educational wording
    # ("I am studying self-harm, but I want to kill myself").
    if any(pattern.search(q) for pattern in CRISIS_ALWAYS) or FIRST_PERSON_DESIRE_TO_DIE.search(q):
        return "crisis"
    personal_context = bool(PERSONAL.search(q))
    speaker = bool(SPEAKER.search(q))
    if speaker and any(pattern.search(q) for pattern in INGESTION_PATTERNS):
        return "medical"

    educational_context = (
        "for a class", "for my class", "for school", "school project", "health class",
        "researching", "studying", "writing about", "reading about", "for an assignment",
        "for my assignment", "for my essay", "for my homework", "for a project",
    )
    # Unambiguous acute symptoms, used to decide whether educational wording excuses a live cue.
    # Broad topic words such as "anaphylaxis" are deliberately absent here, so that "I am
    # researching anaphylaxis for a school project" stays a question.
    live_medical_phrases = (
        "can't breathe", "cant breathe", "cannot breathe", "unable to breathe",
        "struggling to breathe", "gasping", "stopped breathing", "not breathing",
        "isn't breathing", "isnt breathing", "is not breathing", "choking", "collapsed",
        "my throat is closing", "throat is closing", "throat is swelling",
        "took too many", "took too much", "overdosed", "poisoned", "handful of pills",
        "handful of tablets", "my chest hurts", "my chest is tight", "chest hurts",
        "chest is tight", "crushing chest pain", "pain in my chest", "pressure in my chest",
        "having a heart attack", "having a stroke", "face is drooping", "slurred speech",
        "can't speak", "sudden weakness on one side", "severe bleeding", "heavy bleeding",
        "won't stop bleeding", "wont stop bleeding", "bleeding heavily", "bleeding a lot",
        "coughing up blood", "vomiting blood",
        "having a seizure", "having a fit", "convulsion", "unconscious", "unresponsive",
        "passed out", "won't wake", "wont wake", "no pulse", "worst headache", "thunderclap",
    )
    has_live_medical_cue = any(phrase in q for phrase in live_medical_phrases) or any(
        pattern.search(q) for pattern in MEDICAL_PATTERNS
    )
    if any(context in q for context in educational_context) and not has_live_medical_cue:
        return None

    # Tier 2: words that also appear in lessons, poems and news. They only mean an emergency when
    # the question is about a real person rather than a topic.
    if speaker and any(pattern.search(q) for pattern in CRISIS_WITH_MARKER):
        return "crisis"
    if personal_context and CRISIS_DESIRE_TO_DIE.search(q):
        return "crisis"
    if not personal_context:
        return None
    if any(phrase in q for phrase in MEDICAL_PHRASES) or any(p.search(q) for p in MEDICAL_PATTERNS):
        return "medical"
    return None


def is_personal_advice(question):
    q = _normalise(question)
    if any(p.search(q) for p in PERSONAL_ADVICE_PATTERNS):
        return True
    # A number/unit alone is not enough to assume someone is asking about a personal dose.
    has_dose_amount = re.search(r"\b\d+(\.\d+)?\s*(mg|mcg|µg|μg|ml)\b", q)
    return bool(has_dose_amount and PERSONAL.search(q))


def is_harmful_request(question):
    """Return True only for explicit requests that enable serious harm or drug fraud."""
    q = _normalise(question)
    return any(pattern.search(q) for pattern in HARMFUL_REQUEST_PATTERNS)


def check(question):
    """Return a fixed safety action before retrieval, or None for an educational query.

    This is the public entry point used by the pipeline, the evaluation harness, and the FastAPI
    service. Normalization happens here, so all three consumers run the same rules on the same
    normalized text."""
    kind = detect_emergency(question)
    if kind:
        return "emergency", EMERGENCY_RESPONSES[kind]
    if is_harmful_request(question):
        return "refuse", REFUSAL_MESSAGE
    if is_personal_advice(question):
        return "redirect", REDIRECT_MESSAGE
    return None
