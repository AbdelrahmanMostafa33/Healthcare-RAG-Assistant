"""The one pipeline that the notebooks, the evaluation and the API all use.

    question -> safety rules (src/safety.py)
             -> short Gemini safety check for questions not handled by rules
                  emergency/crisis -> fixed message, no retrieval or answer generation
                  redirect/refuse  -> fixed message, no retrieval or answer generation
             -> withhold time-sensitive questions that need a live source
             -> retrieve: FAISS top 40 -> rerank -> up to 5 pages, 2 chunks merged per page
             -> best rerank score below the threshold?  yes -> abstain, no LLM call
                                                         no  -> grounded prompt -> generation (bounded retries for transient 429s)
             -> answer + cited sources + disclaimer
"""
import time

from src import config, safety
from src.deadline import check_deadline
from src.generator import generate, parse_citations
from src.provider_clients import make_generator_client, make_safety_client
from src.retriever import load_retriever
from src.safety_check import SafetyCheckError, classify_for_safety


# Why an answer was withheld. Only INSUFFICIENT_EVIDENCE is a retrieval decision; LIVE_INFORMATION
# is a freshness policy that runs before retrieval, and the others describe a failure or a check
# that never reached an evidence decision. Keeping them separate keeps the metrics honest.
ABSTAIN_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
ABSTAIN_LIVE_INFORMATION = "live_information"
ABSTAIN_CITATION_VALIDATION_FAILED = "citation_validation_failed"
ABSTAIN_UNCALIBRATED = "uncalibrated"
ABSTAIN_SAFETY_CHECK_UNAVAILABLE = "safety_check_unavailable"

DISCLAIMER = (
    "This is general health information for education only. It is not medical advice, "
    "diagnosis or treatment. Talk to a doctor or pharmacist about your own situation, "
    "and call your local emergency number if you think it is an emergency."
)


def _public_source_url(source, url):
    """Hide links known to point to retired NIH pages; the saved passage remains available."""
    publisher = str(source or "").casefold()
    page_url = str(url or "").strip()
    if "nihseniorhealth" in publisher:
        return ""
    if "niddk" in publisher and "/pages/index.aspx" in page_url.casefold():
        return ""
    return page_url

ABSTAIN_MESSAGE = (
    "I can only answer general health-education questions. Your question may be outside that scope, "
    "or my current sources may not contain enough relevant information to answer it reliably. "
    "Try asking a general health question or rephrasing it. If you think this could be an emergency, "
    "call your local emergency number or go to the nearest emergency department now."
)
UNCALIBRATED_MESSAGE = (
    "The assistant is not ready to answer reliably because its evidence check has not been calibrated. "
    "Please try again later."
)
CITATION_CHECK_MESSAGE = (
    "I couldn't verify the sources for my answer, so I withheld it. Please try again later."
)
SAFETY_CHECK_UNAVAILABLE_MESSAGE = (
    "I couldn't complete the safety check. If this may be an emergency, call your local "
    "emergency number now or go to the nearest emergency department."
)


class RAGPipeline:
    def __init__(self, retriever, client, threshold, calibrated, safety_client):
        self.retriever = retriever
        self.client = client
        self.safety_client = safety_client
        self.threshold = threshold
        self.calibrated = calibrated   # False while the placeholder threshold from config is in use

    def answer(self, question, deadline=None):
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        start = time.perf_counter()
        check_deadline(deadline)

        # Keep existing rules first; only unmatched questions reach the short safety model.
        early = safety.check(question)
        if early:
            kind, message = early
            return self._result(
                message, start, emergency=kind == "emergency", redirected=kind == "redirect",
                refused=kind == "refuse",
            )

        check_deadline(deadline)
        try:
            classification = classify_for_safety(self.safety_client, question, deadline=deadline)
        except SafetyCheckError:
            return self._result(
                SAFETY_CHECK_UNAVAILABLE_MESSAGE, start, abstained=True,
                abstain_reason=ABSTAIN_SAFETY_CHECK_UNAVAILABLE,
            )

        action = classification["action"]
        if action in {"emergency_medical", "crisis"}:
            kind = "medical" if action == "emergency_medical" else "crisis"
            return self._result(safety.EMERGENCY_RESPONSES[kind], start, emergency=True)
        if action == "refuse":
            return self._result(safety.REFUSAL_MESSAGE, start, refused=True)
        if action == "redirect":
            return self._result(safety.REDIRECT_MESSAGE, start, redirected=True)

        check_deadline(deadline)
        # A placeholder threshold is not a usable evidence policy. Fail closed until a dev-set
        # threshold is calibrated for this exact model/index/evaluation-data combination.
        if not self.calibrated:
            return self._result(UNCALIBRATED_MESSAGE, start, abstained=True,
                                abstain_reason=ABSTAIN_UNCALIBRATED)

        if safety.requires_live_information(question):
            return self._result(ABSTAIN_MESSAGE, start, abstained=True,
                                abstain_reason=ABSTAIN_LIVE_INFORMATION)

        # FAISS finds candidates first; the cross-encoder then ranks the evidence for this question.
        check_deadline(deadline)
        retrieval_started = time.perf_counter()
        passages = self.retriever.retrieve(question)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        top_score = passages[0]["rerank_score"] if passages else None

        if top_score is None or top_score < self.threshold:
            return self._result(ABSTAIN_MESSAGE, start, abstained=True, top_score=top_score,
                                abstain_reason=ABSTAIN_INSUFFICIENT_EVIDENCE,
                                retrieval_ms=retrieval_ms, retrieved_passages=passages)

        check_deadline(deadline)
        generation_started = time.perf_counter()
        answer = generate(self.client, question, passages, deadline=deadline)
        generation_ms = (time.perf_counter() - generation_started) * 1000
        check_deadline(deadline)
        # Reject the whole answer when any citation is impossible, so an invalid source number can
        # never be displayed next to valid ones. Structural only: a valid ID proves the passage was
        # supplied, not that it supports the claim.
        citations = parse_citations(answer, len(passages))
        if citations["invalid"] or not citations["cited"]:
            return self._result(CITATION_CHECK_MESSAGE, start, abstained=True, top_score=top_score,
                                citation_validation_failed=True,
                                abstain_reason=ABSTAIN_CITATION_VALIDATION_FAILED,
                                invalid_citations=citations["invalid"],
                                retrieval_ms=retrieval_ms, generation_ms=generation_ms,
                                retrieved_passages=passages)
        return self._result(answer, start, passages=passages,
                            cited_numbers=citations["cited"], top_score=top_score,
                            retrieval_ms=retrieval_ms, generation_ms=generation_ms)

    @staticmethod
    def _result(answer, start, abstained=False, emergency=False, redirected=False, refused=False,
                passages=(), top_score=None, citation_validation_failed=False, cited_numbers=None,
                abstain_reason=None, retrieval_ms=None, generation_ms=None,
                retrieved_passages=None, invalid_citations=()):
        passages = list(passages)
        # Retrieved passages are kept separately from the passages a generated answer cites, so an
        # abstention or a rejected answer still records the evidence it actually retrieved.
        retrieved = list(passages if retrieved_passages is None else retrieved_passages)
        cited_numbers = list(range(1, len(passages) + 1)) if cited_numbers is None else cited_numbers
        return {
            "answer": answer,
            "abstained": abstained,     # no generated answer was returned to the user
            "abstain_reason": abstain_reason,   # why, when it was withheld by this pipeline
            "emergency": emergency,     # handled by the emergency message
            "redirected": redirected,   # personal-advice question, handled by the redirect message
            "refused": refused,         # harmful requests are refused before retrieval or generation
            "citation_validation_failed": citation_validation_failed,
            "invalid_citations": list(invalid_citations),   # debug/evaluation only, never shown
            # Only list sources the answer actually cites; n keeps its prompt citation number.
            "sources": [{"n": n, "source": passages[n - 1]["source"], "title": passages[n - 1]["title"],
                         "url": _public_source_url(passages[n - 1]["source"], passages[n - 1]["url"])}
                        for n in cited_numbers],
            "passages": passages,
            "retrieved_passages": retrieved,
            "passage_count": len(retrieved),
            "top_score": top_score,
            "retrieval_ms": round(retrieval_ms, 1) if retrieval_ms is not None else None,
            "generation_ms": round(generation_ms, 1) if generation_ms is not None else None,
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
    generator_client = make_generator_client()   # before the slow model loading, so a missing API key fails fast
    safety_client = make_safety_client()
    threshold, calibrated = config.load_threshold()
    return RAGPipeline(load_retriever(), generator_client, threshold, calibrated, safety_client)
