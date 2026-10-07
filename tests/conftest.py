"""Small fakes so the tests need no model download, GPU or API key."""
import re
import types
import zlib

import faiss
import numpy as np
import pandas as pd
import pytest

from src.pipeline import RAGPipeline
from src.retriever import Retriever

DIM = 64


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

    def predict(self, pairs):
        return np.array([float(len(set(words(query)) & set(words(text)))) for query, text in pairs])


class FakeClient:
    """Stands in for the OpenAI client and records every call."""

    def __init__(self, answer="Diabetes affects blood sugar [1]."):
        self.answer = answer
        self.calls = []
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = types.SimpleNamespace(content=self.answer)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message, finish_reason="stop")])


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
