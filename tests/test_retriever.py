from src import config
from tests.conftest import make_chunks, make_retriever


def test_search_returns_retrieve_k_chunks_with_their_metadata(retriever):
    results = retriever.search("diabetes insulin")
    assert len(results) == config.RETRIEVE_K
    assert {"chunk_id", "doc_id", "source", "title", "url", "text", "faiss_score"} <= set(results[0])
    assert results[0]["chunk_id"] == 0   # the only chunk about diabetes
    scores = [r["faiss_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_puts_the_best_match_first_and_keeps_context_k(retriever):
    reranked = retriever.rerank("topic7 filler7", retriever.search("topic7 filler7"))
    assert len(reranked) == config.CONTEXT_K
    assert reranked[0]["chunk_id"] == 7
    scores = [r["rerank_score"] for r in reranked]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_returns_only_the_chunks_that_go_into_the_prompt(retriever):
    results = retriever.retrieve("diabetes insulin")
    assert len(results) == config.CONTEXT_K
    assert len({r["chunk_id"] for r in results}) == config.CONTEXT_K
    assert all("rerank_score" in r for r in results)


def test_rerank_of_nothing_is_empty(retriever):
    assert retriever.rerank("anything", []) == []


def test_small_index_returns_every_chunk_once_not_a_wrapped_around_row():
    # FAISS pads with -1 when it has fewer chunks than asked for; -1 must not select the last row
    small = make_retriever(make_chunks(3))
    results = small.search("diabetes insulin")
    assert sorted(r["chunk_id"] for r in results) == [0, 1, 2]
