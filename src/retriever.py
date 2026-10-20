"""Retrieve, rerank, and group complementary chunks without repeating copied passages."""
from difflib import SequenceMatcher
import re

import numpy as np

from src import config
from src.kb import load_encoder, load_index

MAX_CHUNKS_PER_SOURCE = 2
NEAR_DUPLICATE_THRESHOLD = 0.88
RERANK_BATCH_SIZE = 16


def _content_tokens(chunk):
    """Ignore the repeated question header when comparing copied answer passages."""
    text = str(chunk.get("text") or "")
    body = text.split("\n", 1)[-1]
    return re.findall(r"[a-z0-9]+", body.casefold())


def _is_near_duplicate(tokens, existing_passages):
    return bool(tokens) and any(
        SequenceMatcher(None, tokens, existing, autojunk=False).ratio()
        >= NEAR_DUPLICATE_THRESHOLD
        for existing in existing_passages
    )


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
        scores = self.reranker.predict(
            [(query, c["text"]) for c in candidates],
            batch_size=RERANK_BATCH_SIZE,
            show_progress_bar=False,
        )
        ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
        selected, selected_tokens, source_counts, source_positions = [], [], {}, {}
        for score, chunk in ranked:
            url = str(chunk.get("url") or "").strip().casefold()
            # The URL identifies a page even if two datasets label its publisher differently.
            source_key = url if url else (chunk.get("source"), chunk.get("doc_id"))
            if source_counts.get(source_key, 0) >= MAX_CHUNKS_PER_SOURCE:
                continue
            tokens = _content_tokens(chunk)
            if _is_near_duplicate(tokens, selected_tokens):
                continue
            if source_key in source_positions:
                # Keep related chunks together so one citation can support a fuller answer.
                selected[source_positions[source_key]]["text"] += f"\n\n{chunk['text']}"
            else:
                if len(selected) == config.CONTEXT_K:
                    continue
                source_positions[source_key] = len(selected)
                selected.append({**chunk, "rerank_score": float(score)})
            selected_tokens.append(tokens)
            source_counts[source_key] = source_counts.get(source_key, 0) + 1
        return selected

    def retrieve(self, query):
        return self.rerank(query, self.search(query))


def load_reranker():
    import torch
    from sentence_transformers import CrossEncoder

    # Half precision reduces memory use and improves throughput on the project's CUDA path.
    use_cuda = torch.cuda.is_available()
    model_kwargs = {"torch_dtype": torch.float16} if use_cuda else None
    # Identity keeps raw logits; recalibrate after changing model precision or retrieval settings.
    return CrossEncoder(
        config.RERANKER_MODEL,
        revision=config.RERANKER_MODEL_REVISION,
        device="cuda" if use_cuda else "cpu",
        model_kwargs=model_kwargs,
        max_length=512,
        activation_fn=torch.nn.Identity(),
    )


def load_retriever():
    index, chunks = load_index()   # first: a missing index should fail before the models load
    return Retriever(index, chunks, load_encoder(), load_reranker())
