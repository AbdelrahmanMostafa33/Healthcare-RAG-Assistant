"""Small fakes so the tests need no model download, GPU or API key."""
import json
import re
import types
import zlib

import faiss
import numpy as np
import pandas as pd
import pytest

from src.pipeline import RAGPipeline
from src.retriever import Retriever
from src import config

DIM = 64


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    """The tests must pass without a key, even when a real one is in the developer's .env."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(config, "GENERATOR_PROVIDER", "gemini")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "groq")


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class FakeEncoder:
    """Hashes words into a bag-of-words vector."""

    def encode(self, texts, **kwargs):
        vectors = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in words(text):
                vectors[i, zlib.crc32(word.encode()) % DIM] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms


class FakeReranker:
    """Score = number of distinct query words found in the text."""

    def predict(self, pairs, batch_size=32, show_progress_bar=None):
        return np.array([float(len(set(words(query)) & set(words(text)))) for query, text in pairs])


class FakeClient:
    """Stands in for the Gemini generation client and records every call."""

    def __init__(self, answer="Diabetes affects blood sugar [1].", finish_reason="stop"):
        self.answer = answer
        self.finish_reason = finish_reason
        self.calls = []
        self.models = types.SimpleNamespace(generate_content=self.generate_content)

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        candidate = types.SimpleNamespace(finish_reason=self.finish_reason)
        return types.SimpleNamespace(text=self.answer, candidates=[candidate])


class FakeJudge:
    """Stand in for the RAG judge and record calls for resume checks."""

    model = "fake-judge"

    def __init__(self):
        self.calls = []

    def __call__(self, system, user, max_tokens=700):
        self.calls.append("answer")
        payload = json.loads(user)
        passages = payload.get("retrieved_passages", [])
        expected_answer = payload.get("expected_behavior") == "answer"
        retrieval_performed = payload.get("retrieval_performed", False)
        generated = payload.get("generated_answer", False)
        if not expected_answer:
            context_verdict = "not_applicable"
        elif passages:
            context_verdict = "pass"
        else:
            context_verdict = "fail" if retrieval_performed else "not_applicable"
        answer_verdict = "pass" if generated else "not_applicable"
        return {
            "context_relevance": {"verdict": context_verdict, "reason": "fake context check"},
            "faithfulness": {"verdict": answer_verdict, "reason": "fake faithfulness check"},
            "answer_relevance": {"verdict": answer_verdict, "reason": "fake relevance check"},
            "behavior": {"verdict": "pass", "reason": "ok"},
        }


def make_chunks(n=25):
    rows = []
    for i in range(n):
        topic = "diabetes insulin sugar" if i == 0 else f"topic{i} filler{i}"
        rows.append({
            "chunk_id": i, "doc_id": i // 2, "source": "MedlinePlus" if i % 2 else "MedQuAD (NIDDK)",
            "title": f"Title {i}", "url": f"https://example.org/{i}",
            "text": f"What is {topic}?\nThis chunk talks about {topic}.",
        })
    return pd.DataFrame(rows)


def make_retriever(chunks):
    encoder = FakeEncoder()
    index = faiss.IndexFlatIP(DIM)
    index.add(encoder.encode(chunks["text"].tolist()))
    return Retriever(index, chunks, encoder, FakeReranker())


@pytest.fixture
def retriever():
    return make_retriever(make_chunks())


@pytest.fixture
def make_pipeline(retriever):
    def make(threshold=0.0, answer="Diabetes affects blood sugar [1]."):
        client = FakeClient(answer)
        return RAGPipeline(retriever, client, threshold, calibrated=True), client
    return make
