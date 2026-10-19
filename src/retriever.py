"""Retrieve the closest chunks, rerank them, and limit repeated context from one source page."""
import numpy as np

from src import config
from src.kb import load_encoder, load_index

MAX_CHUNKS_PER_SOURCE = 2


class Retriever:
    def __init__(self, index, chunks, encoder, reranker):
        self.index = index
        self.chunks = chunks
        self.encoder = encoder
        self.reranker = reranker

    def search(self, query):
        """Dense search: the RETRIEVE_K chunks whose vectors are closest to the query."""
        vector = self.encoder.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, ids = self.index.search(vector, config.RETRIEVE_K)
        # FAISS returns -1 when the index has fewer chunks than asked for
        return [{**self.chunks.iloc[i].to_dict(), "faiss_score": float(score)}
                for score, i in zip(scores[0], ids[0]) if i >= 0]

    def rerank(self, query, candidates):
        """Rerank chunks, then cap each source page so one long page cannot fill the context."""
        if not candidates:
            return []
        scores = self.reranker.predict([(query, c["text"]) for c in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
        selected, source_counts = [], {}
        for score, chunk in ranked:
            url = str(chunk.get("url") or "").strip().casefold()
            source_key = (chunk.get("source"), url) if url else (chunk.get("source"), chunk.get("doc_id"))
            if source_counts.get(source_key, 0) >= MAX_CHUNKS_PER_SOURCE:
                continue
            selected.append({**chunk, "rerank_score": float(score)})
            source_counts[source_key] = source_counts.get(source_key, 0) + 1
            if len(selected) == config.CONTEXT_K:
                break
        return selected

    def retrieve(self, query):
        return self.rerank(query, self.search(query))


def load_reranker():
    import torch
    from sentence_transformers import CrossEncoder

    # Identity keeps the raw logits, so the abstention threshold means the same thing in every library version
    return CrossEncoder(
        config.RERANKER_MODEL,
        revision=config.RERANKER_MODEL_REVISION,
        max_length=512,
        activation_fn=torch.nn.Identity(),
    )


def load_retriever():
    index, chunks = load_index()   # first: a missing index should fail before the models load
    return Retriever(index, chunks, load_encoder(), load_reranker())
