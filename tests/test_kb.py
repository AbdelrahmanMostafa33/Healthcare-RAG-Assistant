import numpy as np
import pandas as pd
import pytest

from src import config, kb


# --- cleaning ---

def test_clean_text_removes_html_tags_but_keeps_clinical_symbols():
    assert kb.clean_text("<p>Take  <b>care</b></p>") == "Take care"
    assert kb.clean_text("dose < 5 mg and BP <120") == "dose < 5 mg and BP <120"


def test_strip_boilerplate_cuts_at_first_marker():
    assert kb.strip_boilerplate("Real answer. Top of Page Related Pages junk") == "Real answer."


def test_html_to_text_parses_lists_and_entities():
    assert kb.html_to_text("<ul><li>Fever</li><li>Cough &amp; cold</li></ul>") == "Fever Cough & cold"


def raw_medquad():
    return pd.DataFrame({
        "document_source": ["NIDDK", "CDC", "NIDDK", "NIDDK", "NIDDK"],
        "document_url": ["https://a", "https://b", "https://c", "https://d", None],
        "question_focus": ["Diabetes", "", "Flu", "Asthma", "Gout"],
        "question": ["What is Diabetes ?", "What is the flu?", "What is Flu?", "What is (are) ?", "What is Gout?"],
        "answer": ["Diabetes is a disease. More on: junk", "The flu is a virus.", "", "Some answer here.", "Gout is arthritis."],
    })


def test_prepare_medquad_filters_rows_and_maps_columns():
    docs = kb.prepare_medquad(raw_medquad())
    assert set(docs["question"]) == {"What is Diabetes?", "What is the flu?", "What is Gout?"}   # empty answer and "(are) ?" dropped
    diabetes = docs[docs["question"] == "What is Diabetes?"].iloc[0]
    assert (diabetes["source"], diabetes["title"], diabetes["url"]) == ("MedQuAD (NIDDK)", "Diabetes", "https://a")
    assert diabetes["text"] == "Diabetes is a disease."
    assert docs[docs["question"] == "What is the flu?"].iloc[0]["title"] == "What is the flu?"   # no focus: title = question
    assert docs[docs["question"] == "What is Gout?"].iloc[0]["url"] == ""


def test_prepare_medquad_keeps_the_longest_answer_for_a_repeated_question():
    raw = pd.DataFrame({
        "document_source": ["A", "A"], "document_url": ["u1", "u2"], "question_focus": ["X", "X"],
        "question": ["What is X?", "What is X?"], "answer": ["short answer", "a much longer answer text"],
    })
    docs = kb.prepare_medquad(raw)
    assert len(docs) == 1 and docs.iloc[0]["text"] == "a much longer answer text"


def test_prepare_medquad_rejects_the_wrong_dataset_version():
    with pytest.raises(ValueError, match="lavita/MedQuAD"):
        kb.prepare_medquad(pd.DataFrame({"qtype": ["symptoms"], "Question": ["q?"], "Answer": ["a"]}))


MEDLINEPLUS_XML = b"""<health-topics>
  <health-topic title="Asthma" url="https://medlineplus.gov/asthma.html" language="English">
    <full-summary>&lt;p&gt;Asthma is a lung disease.&lt;/p&gt;</full-summary>
  </health-topic>
  <health-topic title="Asma" url="https://medlineplus.gov/spanish/asthma.html" language="Spanish">
    <full-summary>&lt;p&gt;Es una enfermedad.&lt;/p&gt;</full-summary>
  </health-topic>
</health-topics>"""


def test_medlineplus_to_kb_table():
    topics = kb.parse_medlineplus_xml(MEDLINEPLUS_XML)
    assert topics["title"].tolist() == ["Asthma"]   # the Spanish topic is skipped
    medline = kb.prepare_medlineplus(topics)
    assert medline.iloc[0]["text"] == "Asthma is a lung disease."
    assert medline.iloc[0]["question"] == "What is Asthma?"
    table = kb.build_kb(kb.prepare_medquad(raw_medquad()), medline)
    assert table["doc_id"].tolist() == list(range(len(table)))
    assert list(table.columns) == kb.KB_COLUMNS


# --- chunking ---

def sentences(n):
    return " ".join(f"Sentence number {i} " + "word " * 8 + "ends here." for i in range(n))


@pytest.fixture
def small_chunks(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_MAX_CHARS", 300)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_CHARS", 100)


def test_split_text_respects_the_size_limit_and_keeps_every_sentence(small_chunks):
    chunks = kb.split_text(sentences(40))
    assert len(chunks) > 1 and all(len(c) <= 300 for c in chunks)
    for i in range(40):
        assert any(f"Sentence number {i} " in c for c in chunks)


def test_split_text_repeats_the_last_sentence_at_the_start_of_the_next_chunk(small_chunks):
    first, second = kb.split_text(sentences(10))[:2]
    last_sentence = first.split(". ")[-1].rstrip(".") if ". " in first else first
    assert last_sentence in second


def test_split_text_without_punctuation_splits_on_words(small_chunks):
    text = "symptoms include " + " - ".join(["a very long list item"] * 80)
    chunks = kb.split_text(text)
    assert len(chunks) > 1 and all(len(c) <= 300 for c in chunks)


def docs_for_chunking():
    return pd.DataFrame([
        {"doc_id": 0, "source": "MedQuAD (NIDDK)", "title": "Diabetes", "url": "https://x",
         "question": "What is Diabetes?", "text": sentences(30)},
        {"doc_id": 1, "source": "MedlinePlus", "title": "Tiny", "url": "https://y",
         "question": "What is Tiny?", "text": "ok"},
    ])


def test_every_chunk_has_the_question_on_top_and_real_content_below(small_chunks):
    chunks = kb.build_chunks(docs_for_chunking())
    assert chunks["chunk_id"].tolist() == list(range(len(chunks)))
    assert set(chunks["doc_id"]) == {0}   # the one-word document is too short to become a chunk
    for text in chunks["text"]:
        header, _, body = text.partition("\n")
        assert header == "What is Diabetes?" and len(body) >= kb.MIN_CHUNK_CHARS


def test_chunks_without_question_contain_only_the_body(small_chunks):
    chunks = kb.build_chunks(docs_for_chunking(), with_question=False)
    assert not chunks["text"].str.contains("What is Diabetes").any()


# --- index ---

def index_and_chunks(n=6, dim=8):
    vectors = np.random.default_rng(0).normal(size=(n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    chunks = pd.DataFrame({"chunk_id": range(n), "doc_id": range(n), "source": "MedlinePlus",
                           "title": [f"T{i}" for i in range(n)], "url": "", "text": [f"text {i}" for i in range(n)]})
    return kb.build_index(vectors), chunks, vectors


@pytest.fixture
def index_files(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_PATH", tmp_path / "kb.faiss")
    monkeypatch.setattr(config, "CHUNKS_PATH", tmp_path / "chunks.jsonl")


def test_index_and_chunk_table_survive_a_save_and_load(index_files):
    index, chunks, vectors = index_and_chunks()
    kb.save_index(index, chunks)
    loaded_index, loaded_chunks = kb.load_index()
    assert loaded_index.ntotal == len(loaded_chunks) == 6
    assert loaded_chunks["title"].tolist() == chunks["title"].tolist()
    assert loaded_chunks["url"].tolist() == [""] * 6
    assert loaded_index.search(vectors[2:3], 1)[1][0][0] == 2


def test_index_and_chunk_table_must_have_the_same_length(index_files):
    index, chunks, _ = index_and_chunks()
    with pytest.raises(ValueError, match="vectors"):
        kb.save_index(index, chunks.iloc[:-1])


def test_loading_a_missing_index_says_how_to_build_it(index_files):
    with pytest.raises(FileNotFoundError, match="notebooks 01 and 02"):
        kb.load_index()
