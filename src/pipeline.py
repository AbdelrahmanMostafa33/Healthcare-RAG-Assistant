"""The one pipeline that the notebooks, the evaluation and the API all use.

    question -> safety check (src/safety.py)
                  emergency        -> fixed emergency message, no retrieval, no LLM
                  personal advice  -> fixed redirect to a doctor, no retrieval, no LLM
                  harmful request  -> fixed refusal message, no retrieval, no LLM
             -> retrieve: FAISS top 40 -> rerank -> up to 5, one per source page
             -> best rerank score below the threshold?  yes -> abstain, no LLM call
                                                         no  -> grounded prompt -> generation (bounded retries for transient 429s)
             -> answer + cited sources + disclaimer
"""
import time

from src import config, safety
from src.generator import citation_numbers, citations_are_valid, generate, make_generator_client
from src.retriever import load_retriever

# Why an answer was withheld. Only INSUFFICIENT_EVIDENCE is a retrieval decision; the other two are
# not statements about the sources, and counting them as abstentions distorts abstention metrics.
ABSTAIN_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
ABSTAIN_CITATION_VALIDATION_FAILED = "citation_validation_failed"
ABSTAIN_UNCALIBRATED = "uncalibrated"

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

        # Emergencies and personal-advice questions never reach retrieval or the LLM.
        early = safety.check(question)
        if early:
            kind, message = early
            return self._result(
                message, start, emergency=kind == "emergency", redirected=kind == "redirect",
                refused=kind == "refuse",
            )

        # A placeholder threshold is not a usable evidence policy. Fail closed until a dev-set
        # threshold is calibrated for this exact model/index/evaluation-data combination.
        if not self.calibrated:
            return self._result(UNCALIBRATED_MESSAGE, start, abstained=True,
                                abstain_reason=ABSTAIN_UNCALIBRATED)

        # FAISS finds candidates first; the cross-encoder then ranks the evidence for this question.
        retrieval_started = time.perf_counter()
        passages = self.retriever.retrieve(question)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        top_score = passages[0]["rerank_score"] if passages else None

        if top_score is None or top_score < self.threshold:
            return self._result(ABSTAIN_MESSAGE, start, abstained=True, top_score=top_score,
                                abstain_reason=ABSTAIN_INSUFFICIENT_EVIDENCE,
                                retrieval_ms=retrieval_ms, retrieved_passages=passages)

        generation_started = time.perf_counter()
        answer = generate(self.client, question, passages)
        generation_ms = (time.perf_counter() - generation_started) * 1000
        if not citations_are_valid(answer, passages):
            # Do not expose an answer with missing or impossible source references.
            return self._result(CITATION_CHECK_MESSAGE, start, abstained=True, top_score=top_score,
                                citation_validation_failed=True,
                                abstain_reason=ABSTAIN_CITATION_VALIDATION_FAILED,
                                retrieval_ms=retrieval_ms, generation_ms=generation_ms,
                                retrieved_passages=passages)
        return self._result(answer, start, passages=passages,
                            cited_numbers=citation_numbers(answer, len(passages)), top_score=top_score,
                            retrieval_ms=retrieval_ms, generation_ms=generation_ms)

    @staticmethod
    def _result(answer, start, abstained=False, emergency=False, redirected=False, refused=False,
                passages=(), top_score=None, citation_validation_failed=False, cited_numbers=None,
                abstain_reason=None, retrieval_ms=None, generation_ms=None,
                retrieved_passages=None):
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
    client = make_generator_client()   # before the slow model loading, so a missing API key fails fast
    threshold, calibrated = config.load_threshold()
    return RAGPipeline(load_retriever(), client, threshold, calibrated)
