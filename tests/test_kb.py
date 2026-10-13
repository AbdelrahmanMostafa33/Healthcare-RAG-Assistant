import numpy as np
import pandas as pd
import pytest

from src import config, kb


# --- cleaning ---

def test_clean_text_removes_html_tags_but_keeps_clinical_symbols():
    assert kb.clean_text("<p>Take  <b>care</b></p>") == "Take care"
    assert kb.clean_text("dose < 5 mg and BP <120") == "dose < 5 mg and BP <120"


def test_strip_boilerplate_drops_a_trailing_navigation_block():
    assert kb.strip_boilerplate("Real answer. Top of Page Related Pages junk") == "Real answer."


def test_strip_boilerplate_keeps_a_marker_inside_the_answer():
    """A mid-answer marker introduces more content; 57 raw answers lost text to the old rule."""
    text = "Treatment works. Go to: the emergency room if the pain persists."
    assert kb.strip_boilerplate(text) == text
    # The real shape: the marker sits early, and the answer keeps going after it.
    long_answer = "More on: Prevention & Control A lice comb helps. " + "More detail about lice follows. " * 40
    assert "More on:" in kb.strip_boilerplate(long_answer)


def test_strip_boilerplate_cuts_a_marker_in_the_last_quarter_even_if_a_sentence_follows():
    """The copyright notice after 'Top of Page' is boilerplate even though it is prose."""
    text = "Vaccines protect children. " * 20 + "Top of Page Images and logos on this website are not public domain."
    assert "Top of Page" not in kb.strip_boilerplate(text)


def test_strip_boilerplate_reports_how_much_it_removed():
    removed = []
    kb.strip_boilerplate("Real answer. Top of Page junk", removed_stats=removed)
    assert removed == [len("Top of Page junk")]
    kb.strip_boilerplate("An answer with no marker at all.", removed_stats=removed)
    assert removed == [len("Top of Page junk")]   # nothing dropped, nothing recorded


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


def test_prepare_medquad_retains_answer_bearing_gard_rows():
    raw = pd.DataFrame({
        "document_source": ["GARD"], "document_url": ["https://rarediseases.info.nih.gov/gard/1/example"],
        "question_focus": ["Example condition"], "question": ["What is Example condition?"],
        "answer": ["Example condition is a rare disease."],
    })
    docs = kb.prepare_medquad(raw)
    assert len(docs) == 1 and docs.iloc[0]["source"] == "MedQuAD (GARD)"


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


# --- MedQuAD/MedlinePlus deduplication ---

def test_medlineplus_slug_reads_both_url_forms():
    """MedQuAD links to the old host; the official XML uses medlineplus.gov."""
    assert kb.medlineplus_slug("https://www.nlm.nih.gov/medlineplus/DiabetesType2.html") == "diabetestype2"
    assert kb.medlineplus_slug("https://medlineplus.gov/diabetestype2.html") == "diabetestype2"
    assert kb.medlineplus_slug("") == ""


def test_build_kb_drops_a_medquad_topic_the_official_medlineplus_xml_already_has():
    medquad = pd.DataFrame({
        "source": ["MedQuAD (MPlusHealthTopics)", "MedQuAD (MPlusHealthTopics)", "MedQuAD (NIDDK)"],
        "title": ["Asthma", "Gout", "Diabetes"],
        "url": ["https://www.nlm.nih.gov/medlineplus/asthma.html",
                "https://www.nlm.nih.gov/medlineplus/gout.html",
                "https://www.niddk.nih.gov/diabetes"],
        "question": ["What is Asthma?", "What is Gout?", "What is Diabetes?"],
        "text": ["Asthma answer.", "Gout answer.", "Diabetes answer."],
    })
    medline = pd.DataFrame({
        "source": ["MedlinePlus"], "title": ["Asthma"], "url": ["https://medlineplus.gov/asthma.html"],
        "question": ["What is Asthma?"], "text": ["Asthma summary."],
    })
    table = kb.build_kb(medquad, medline)
    questions = table["question"].tolist()
    assert questions.count("What is Asthma?") == 1          # the duplicate MedQuAD row is gone
    assert table.loc[table["question"] == "What is Asthma?", "source"].tolist() == ["MedlinePlus"]
    assert "What is Gout?" in questions                     # no XML twin: MedQuAD row is kept
    assert "What is Diabetes?" in questions                 # a non-MedlinePlus publisher is never touched
    assert table["doc_id"].tolist() == list(range(len(table)))


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
        assert len(text) <= config.CHUNK_MAX_CHARS


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
    monkeypatch.setattr(config, "INDEX_MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(config, "KB_PATH", tmp_path / "kb_documents.csv")


def test_index_and_chunk_table_survive_a_save_and_load(index_files):
    index, chunks, vectors = index_and_chunks()
    kb.save_index(index, chunks)
    manifest = __import__("json").loads(config.INDEX_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["embedding_model_revision"] == config.EMBEDDING_MODEL_REVISION
    assert manifest["reranker_model_revision"] == config.RERANKER_MODEL_REVISION
    assert manifest["model_runtime_versions"] == config.model_runtime_versions()
    loaded_index, loaded_chunks = kb.load_index()
    assert loaded_index.ntotal == len(loaded_chunks) == 6
    assert loaded_chunks["title"].tolist() == chunks["title"].tolist()
    assert loaded_chunks["url"].tolist() == [""] * 6
    assert loaded_index.search(vectors[2:3], 1)[1][0][0] == 2


def test_loading_warns_instead_of_rejecting_a_library_version_change(index_files, caplog):
    """A torch patch release must not brick a built index; a model or revision change still does."""
    index, chunks, _ = index_and_chunks()
    kb.save_index(index, chunks)
    manifest = __import__("json").loads(config.INDEX_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["model_runtime_versions"] = {"sentence-transformers": "0.0.0"}
    config.INDEX_MANIFEST_PATH.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    loaded_index, loaded_chunks = kb.load_index()
    assert loaded_index.ntotal == len(loaded_chunks) == 6
    assert "different library versions" in caplog.text


def test_loading_rejects_a_changed_processed_knowledge_base(index_files, tmp_path, monkeypatch):
    config.KB_PATH.write_text("doc_id,text\n0,original\n", encoding="utf-8")
    index, chunks, _ = index_and_chunks()
    kb.save_index(index, chunks)
    config.KB_PATH.write_text("doc_id,text\n0,changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Processed knowledge base changed"):
        kb.load_index()


def test_loading_rejects_an_index_built_with_a_different_reranker(index_files):
    index, chunks, _ = index_and_chunks()
    kb.save_index(index, chunks)
    manifest = __import__("json").loads(config.INDEX_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["reranker_model"] = "different/reranker"
    config.INDEX_MANIFEST_PATH.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="different model/retrieval settings"):
        kb.load_index()


def test_loading_rejects_an_index_built_with_a_different_model_revision(index_files):
    index, chunks, _ = index_and_chunks()
    kb.save_index(index, chunks)
    manifest = __import__("json").loads(config.INDEX_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["embedding_model_revision"] = "a-different-snapshot"
    config.INDEX_MANIFEST_PATH.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="different model/retrieval settings"):
        kb.load_index()


def test_index_and_chunk_table_must_have_the_same_length(index_files):
    index, chunks, _ = index_and_chunks()
    with pytest.raises(ValueError, match="vectors"):
        kb.save_index(index, chunks.iloc[:-1])


def test_loading_a_missing_index_says_how_to_build_it(index_files):
    with pytest.raises(FileNotFoundError, match="notebooks 01 and 02"):
        kb.load_index()


def test_build_index_rejects_empty_or_non_finite_embeddings():
    with pytest.raises(ValueError, match="non-empty 2D"):
        kb.build_index(np.empty((0, 8), dtype=np.float32))
    bad = np.ones((2, 3), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        kb.build_index(bad)
