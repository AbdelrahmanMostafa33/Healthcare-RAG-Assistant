"""The one pipeline that the notebooks, the evaluation and the API all use.

    question -> retrieve (FAISS top 20 -> rerank -> top 5)
             -> best rerank score below the threshold?  yes -> abstain, no LLM call
                                                         no  -> grounded prompt -> one LLM call
             -> answer + the sources that were in the prompt + disclaimer
"""
import time

from src import config
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
            # the passages that were in the prompt; n matches the [n] citations in the answer
            "sources": [{"n": n, "source": p["source"], "title": p["title"], "url": p["url"]}
                        for n, p in enumerate(passages, start=1)],
            "passages": passages,
            "top_score": top_score,
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
