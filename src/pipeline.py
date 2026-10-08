"""The one pipeline that the notebooks, the evaluation and the API all use.

    question -> emergency check -> out-of-scope check -> retrieve
             -> best rerank score below the threshold?  yes -> abstain, no LLM call
                                                         no  -> grounded prompt -> one LLM call
             -> answer + the sources that were in the prompt + disclaimer

Three checks run before retrieval, in order:

1. Emergency (chest pain, stroke signs, ...) -> fixed emergency response.
2. Medication (named drugs, doses, interactions) -> fixed out-of-scope message.
3. Personal advice (own symptoms, own results, "should I take", ...) ->
   fixed redirect to a doctor.

The first two are matched by keyword; the third too. The point is that a static
consumer-health knowledge base cannot answer these, so saying "I couldn't find
information" is the wrong message — a redirect is.
"""
import re
import time

from src import config
from src.emergency import detect_emergency, emergency_response
from src.generator import generate, make_client
from src.retriever import load_retriever

DISCLAIMER = (
    "This is general health information for education only. It is not medical advice, "
    "diagnosis or treatment. Talk to a doctor or pharmacist about your own situation, "
    "and call your local emergency number if you think it is an emergency."
)

ABSTAIN_MESSAGE = (
    "I couldn't find enough relevant information in my current sources to answer this reliably. "
    "Try rephrasing the question, or ask a doctor or pharmacist. If you think this could be an "
    "emergency, call your local emergency number or go to the nearest emergency department now."
)

MEDICATION_REDIRECT_MESSAGE = (
    "Medicines are outside what this assistant covers. The sources it uses describe "
    "conditions and how they are generally treated, not specific drugs, doses or "
    "interactions. For questions about a specific medicine, ask a doctor or pharmacist. "
    "If you think this could be an emergency, call your local emergency number or go to "
    "the nearest emergency department now."
)

SAFETY_REDIRECT_MESSAGE = (
    "This looks like a question about your own situation or a specific person. This "
    "assistant gives general health information only — it cannot look at your symptoms, "
    "results or circumstances, and it does not give personal medical advice. Please "
    "speak to a doctor or pharmacist about your own situation. If you think this could "
    "be an emergency, call your local emergency number or go to the nearest emergency "
    "department now."
)

# Medicines are out of scope. Anything that names a drug or asks about dosing,
# side effects or interactions is redirected before retrieval.
_MEDICATION_PATTERNS = [
    r"\b\d+\s*(mg|ml|mcg|μg|g)\b",
    r"\bhow (many|much) (mg|ml|mcg|dose)\b",
    r"\b(side effects?|interactions?|dosage)\s+of\b",
    r"\b(metformin|ibuprofen|paracetamol|acetaminophen|amoxicillin|omeprazole|"
    r"prednisone|statin|aspirin|warfarin|levothyroxine|insulin|antihistamine|"
    r"corticosteroid|antibiotic|naproxen|codeine|morphine|opioid)s?\b",
    r"\b(taking|take|takes|took|using|use|uses|used|giving|give|gives|gave)\s+"
    r"(an?\s+|the\s+)?(medicine|medication|drug|pill|antibiotic|prescription)s?\b",
]

# Personal advice cannot come from a static KB either. Anything about the user's
# own situation, their own results, or another specific person is redirected.
_SAFETY_PATTERNS = [
    r"\bmy (child|son|daughter|wife|husband|mother|father|baby|kid|partner|"
    r"brother|sister|grandmother|grandfather|friend|family)\b",
    r"\bmy \d+[- ]?(year|month|yo|yr)s?[- ]?old\b",
    r"\b(should|can|may) i (take|give|use|stop|start|drink|eat|try|apply|inject)\b",
    r"\bwhat (should|do) i do\b",
    r"\bis it safe (to|for)\b",
    r"\b(my|the) (blood|test|scan|x-?ray|mri|ct|ecg|ekg|lab|urine|biopsy) results?\b",
    r"\binterpret (my|these|the)\b",
    r"\bam i (having|dying|pregnant|sick|ok)\b",
]


def detect_out_of_scope(question):
    """Return 'medication', 'safety_redirect' or None.

    Medicines and personal-advice questions can't be answered from a static
    consumer-health knowledge base, so the pipeline redirects them instead of
    abstaining. Order matters: medication first, since questions that are both
    ("is it safe to give my child ibuprofen?") should get the medication message.
    """
    q = question.lower()
    for pattern in _MEDICATION_PATTERNS:
        if re.search(pattern, q):
            return "medication"
    for pattern in _SAFETY_PATTERNS:
        if re.search(pattern, q):
            return "safety_redirect"
    return None


class RAGPipeline:
    def __init__(self, retriever, client, threshold, calibrated):
        self.retriever = retriever
        self.client = client
        self.threshold = threshold
        self.calibrated = calibrated   # False while the placeholder threshold from config is in use

    def answer(self, question):
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        start = time.perf_counter()

        # 1. Emergency. Runs before retrieval so an acute emergency never gets a
        #    general information answer and never reaches the LLM.
        emergency_kind = detect_emergency(question) if config.EMERGENCY_ENABLED else None
        if emergency_kind:
            return self._early_response(emergency_response(emergency_kind), start, kind="emergency")

        # 2. Out of scope. Medicines and personal-advice questions can't come
        #    from the sources, so they are redirected instead of abstained on.
        out_of_scope = detect_out_of_scope(question)
        if out_of_scope == "medication":
            return self._early_response(MEDICATION_REDIRECT_MESSAGE, start, kind="medication")
        if out_of_scope == "safety_redirect":
            return self._early_response(SAFETY_REDIRECT_MESSAGE, start, kind="safety_redirect")

        # 3. Normal retrieval path.
        passages = self.retriever.retrieve(question)
        top_score = passages[0]["rerank_score"] if passages else None

        abstained = top_score is None or top_score < self.threshold
        if abstained:
            answer, passages = ABSTAIN_MESSAGE, []
        else:
            answer = generate(self.client, question, passages)

        return {
            "answer": answer,
            "abstained": abstained,
            "emergency": False,
            "redirect": None,
            # the passages that were in the prompt; n matches the [n] citations in the answer
            "sources": [{"n": n, "source": p["source"], "title": p["title"], "url": p["url"]}
                        for n, p in enumerate(passages, start=1)],
            "passages": passages,
            "top_score": top_score,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "disclaimer": DISCLAIMER,
        }

    @staticmethod
    def _early_response(message, start, kind):
        """Fixed response for a question that never reaches retrieval."""
        return {
            "answer": message,
            "abstained": kind == "medication",   # a medicine question is an abstention, a redirect is not
            "emergency": kind == "emergency",
            "redirect": kind if kind != "emergency" else None,
            "sources": [],
            "passages": [],
            "top_score": None,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "disclaimer": DISCLAIMER,
        }


def format_result(result):
    lines = [result["answer"]]
    if result["sources"]:
        lines.append("\nSources:")
        for s in result["sources"]:
            lines.append(f"{s['n']}. {s['source']} - {s['title']}")
            if s["url"]:
                lines.append(f"   {s['url']}")
    lines.append(f"\n{result['disclaimer']}")
    return "\n".join(lines)


def load_pipeline():
    client = make_client()   # before the slow model loading, so a missing API key fails fast
    threshold, calibrated = config.load_threshold()
    return RAGPipeline(load_retriever(), client, threshold, calibrated)