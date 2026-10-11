"""Build and load the knowledge base: clean the sources, split them into chunks, embed, index."""
import hashlib
import html
import json
import os
import re
from xml.etree import ElementTree

import faiss
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from src import config

# NIH pages end with site navigation text. Everything after the first marker is dropped.
BOILERPLATE = [
    "More on:", "Top of Page", "Related Pages", "Images and logos on this website",
    "This graphic notice means", "Go to:", "Back to Top",
]
TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")   # real tags only, so "<5 mg" survives
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
MIN_CHUNK_CHARS = 30

KB_COLUMNS = ["doc_id", "source", "title", "url", "question", "text"]
MEDQUAD_COLUMNS = {"document_source", "document_url", "question_focus", "question", "answer"}


# --- cleaning -------------------------------------------------------------

def clean_text(text):
    if text is None or pd.isna(text):
        return ""
    text = TAG_RE.sub(" ", str(text))
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+\?", "?", text)
    text = re.sub(r"\?+", "?", text)
    return text.replace(" .", ".").replace(" ,", ",").strip()


def strip_boilerplate(text):
    for marker in BOILERPLATE:
        text = text.split(marker)[0]
    return re.sub(r"\s+", " ", text).strip()


def html_to_text(text):
    """MedlinePlus summaries contain real HTML (lists, links), so use a parser."""
    if text is None or pd.isna(text):
        return ""
    plain = BeautifulSoup(str(text), "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", html.unescape(plain)).strip()


def prepare_medquad(raw):
    """lavita/MedQuAD rows -> one KB row per question/answer pair."""
    missing = MEDQUAD_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"MedQuAD data is missing columns {sorted(missing)}. Use the lavita/MedQuAD dataset.")

    # Rows without answers are unusable. This filters the drug, A.D.A.M. and herbal subsets in
    # the checked snapshot; answer-bearing GARD rare-disease rows are retained.
    raw = raw[raw["answer"].fillna("").str.strip() != ""]

    question = raw["question"].apply(clean_text)
    focus = raw["question_focus"].apply(clean_text)
    publisher = raw["document_source"].fillna("").astype(str).str.strip()
    df = pd.DataFrame({
        "source": publisher.map(lambda p: f"MedQuAD ({p})" if p else "MedQuAD"),
        "title": focus.where(focus != "", question),
        "url": raw["document_url"].fillna("").astype(str),
        "question": question,
        "text": raw["answer"].apply(clean_text).apply(strip_boilerplate),
    })

    df = df[(df["question"].str.len() > 5) & (df["text"] != "")]
    df = df[~df["question"].str.contains(r"\(are\)\s*\?", regex=True)]   # "What is (are) ?" = empty topic name
    # same question twice: keep the longer answer
    df = df.assign(length=df["text"].str.len()).sort_values("length", ascending=False)
    return df.drop_duplicates(subset="question").drop(columns="length").reset_index(drop=True)


def parse_medlineplus_xml(xml_bytes):
    """MedlinePlus topics XML -> title, url, summary (English topics only)."""
    rows = []
    for topic in ElementTree.fromstring(xml_bytes).findall("health-topic"):
        if topic.get("language") != "English":
            continue
        summary = topic.find("full-summary")
        rows.append({
            "title": topic.get("title", ""),
            "url": topic.get("url", ""),
            "summary": "".join(summary.itertext()).strip() if summary is not None else "",
        })
    return pd.DataFrame(rows, columns=["title", "url", "summary"])


def prepare_medlineplus(raw):
    """One KB row per health topic. MedlinePlus has no questions, so we use 'What is <topic>?'."""
    title = raw["title"].apply(clean_text)
    df = pd.DataFrame({
        "source": "MedlinePlus",
        "title": title,
        "url": raw["url"].fillna("").astype(str),
        "question": ("What is " + title + "?").apply(clean_text),
        "text": raw["summary"].apply(html_to_text),
    })
    df = df[(df["text"] != "") & (df["title"] != "")]
    return df.drop_duplicates(subset="url").reset_index(drop=True)


def build_kb(medquad, medlineplus):
    kb = pd.concat([medquad, medlineplus], ignore_index=True)
    kb.insert(0, "doc_id", range(len(kb)))
    return kb[KB_COLUMNS]


# --- chunking -------------------------------------------------------------

def _split_long_sentence(sentence, max_chars):
    """Split a run-on sentence/list on word boundaries when needed."""
    parts, current = [], ""
    for word in sentence.split():
        if len(word) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.extend(word[i:i + max_chars] for i in range(0, len(word), max_chars))
            continue
        if current and len(current) + 1 + len(word) > max_chars:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    return parts + [current]


def split_text(text, max_chars=None):
    """Pack sentences into bounded chunks, repeating a short boundary sentence where possible."""
    max_chars = config.CHUNK_MAX_CHARS if max_chars is None else int(max_chars)
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    sentences = []
    for sentence in SENTENCE_END.split(text.strip()):
        if len(sentence) > max_chars:
            sentences.extend(_split_long_sentence(sentence, max_chars))
        elif sentence:
            sentences.append(sentence)

    chunks, current = [], []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence])) > max_chars:
            chunks.append(" ".join(current))
            last = current[-1]
            fits = len(last) <= config.CHUNK_OVERLAP_CHARS and len(last) + 1 + len(sentence) <= max_chars
            current = [last] if fits else []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunks(kb, with_question=True):
    """One row per chunk. With with_question=True the document's question goes on top of every
    chunk, so a chunk from the middle of a long answer still says what it is about. It is added
    after splitting, so a chunk can never consist of the question alone."""
    rows = []
    for doc in kb.itertuples(index=False):
        raw_question = getattr(doc, "question", "")
        question = "" if pd.isna(raw_question) else str(raw_question).strip()
        if not with_question:
            question = ""
        raw_text = getattr(doc, "text", "")
        body_text = "" if pd.isna(raw_text) else str(raw_text).strip()
        # The topic header is part of the embedding text, so reserve room for it before splitting.
        # Keep the complete question; pathological questions that leave no room for evidence are skipped.
        prefix_chars = len(question) + 1 if question else 0
        body_budget = config.CHUNK_MAX_CHARS - prefix_chars
        if body_budget < MIN_CHUNK_CHARS:
            continue
        for body in split_text(body_text, max_chars=body_budget):
            if len(body) < MIN_CHUNK_CHARS:
                continue
            text = f"{question}\n{body}" if question else body
            if len(text) > config.CHUNK_MAX_CHARS:
                raise AssertionError("chunk exceeds CHUNK_MAX_CHARS")
            rows.append((doc.doc_id, doc.source, doc.title, doc.url, text))
    chunks = pd.DataFrame(rows, columns=["doc_id", "source", "title", "url", "text"])
    chunks.insert(0, "chunk_id", range(len(chunks)))
    return chunks


# --- embeddings and index -------------------------------------------------

def load_encoder():
    import torch
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(config.EMBEDDING_MODEL)
    encoder.max_seq_length = 512
    return encoder.half() if torch.cuda.is_available() else encoder


def encode_texts(encoder, texts, batch_size=32):
    """Unit-length vectors, so inner product = cosine similarity."""
    vectors = encoder.encode(list(texts), batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)


def build_index(embeddings):
    """Build an exact cosine-similarity index from normalized float32 embeddings."""
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty 2D array")
    if not np.isfinite(vectors).all():
        raise ValueError("embeddings contain NaN or infinite values")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError("embeddings contain a zero vector")
    # Avoid silently changing retrieval semantics: inner product is cosine only for unit vectors.
    vectors = vectors / norms[:, None]
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def _check_aligned(index, chunks):
    required = {"chunk_id", "doc_id", "source", "title", "url", "text"}
    missing = required - set(chunks.columns)
    if missing:
        raise ValueError(f"Chunk table is missing columns: {sorted(missing)}")
    if index.ntotal != len(chunks):
        raise ValueError(f"Index has {index.ntotal} vectors but the chunk table has {len(chunks)} rows. Rebuild with notebook 02.")
    if len(chunks) == 0:
        raise ValueError("The index and chunk table are empty. Rebuild the knowledge base and index.")
    if not np.array_equal(chunks["chunk_id"].to_numpy(), np.arange(len(chunks))):
        raise ValueError("chunk_id must equal the row position in the chunk table.")
    if chunks["doc_id"].isna().any():
        raise ValueError("Chunk table contains missing document IDs. Rebuild with notebook 02.")
    for column in ("source", "title"):
        if chunks[column].isna().any() or chunks[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Chunk table contains empty {column} values. Rebuild with notebook 02.")
    if chunks["text"].isna().any() or chunks["text"].astype(str).str.strip().eq("").any():
        raise ValueError("Chunk table contains empty text. Rebuild with notebook 02.")


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_index(index, chunks):
    """Atomically write an index, its ordered chunk table, and a compatibility manifest."""
    _check_aligned(index, chunks)
    config.INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    index_tmp = config.INDEX_PATH.with_name(config.INDEX_PATH.name + ".tmp")
    chunks_tmp = config.CHUNKS_PATH.with_name(config.CHUNKS_PATH.name + ".tmp")
    manifest_tmp = config.INDEX_MANIFEST_PATH.with_name(config.INDEX_MANIFEST_PATH.name + ".tmp")
    try:
        faiss.write_index(index, str(index_tmp))
        chunks.to_json(chunks_tmp, orient="records", lines=True, force_ascii=False)
        os.replace(index_tmp, config.INDEX_PATH)
        os.replace(chunks_tmp, config.CHUNKS_PATH)
        manifest = {
            "embedding_model": config.EMBEDDING_MODEL,
            "reranker_model": config.RERANKER_MODEL,
            "chunk_max_chars": config.CHUNK_MAX_CHARS,
            "chunk_overlap_chars": config.CHUNK_OVERLAP_CHARS,
            "num_chunks": int(len(chunks)),
            "vector_dimension": int(index.d),
            "processed_kb_sha256": config.sha256_file(config.KB_PATH),
            "index_sha256": _file_sha256(config.INDEX_PATH),
            "chunks_sha256": _file_sha256(config.CHUNKS_PATH),
        }
        manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(manifest_tmp, config.INDEX_MANIFEST_PATH)
    finally:
        for path in (index_tmp, chunks_tmp, manifest_tmp):
            if path.exists():
                path.unlink()


def load_index():
    """Load only a complete index build whose manifest, vectors and chunks agree."""
    required = (config.INDEX_PATH, config.CHUNKS_PATH, config.INDEX_MANIFEST_PATH)
    if not all(path.exists() for path in required):
        raise FileNotFoundError("Index files or manifest not found. Run notebooks 01 and 02 to build them.")
    try:
        manifest = json.loads(config.INDEX_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Index manifest is invalid. Rebuild with notebook 02.") from error

    if not isinstance(manifest, dict):
        raise ValueError("Index manifest must contain a JSON object. Rebuild with notebook 02.")
    expected_settings = {
        "embedding_model": config.EMBEDDING_MODEL,
        "reranker_model": config.RERANKER_MODEL,
        "chunk_max_chars": config.CHUNK_MAX_CHARS,
        "chunk_overlap_chars": config.CHUNK_OVERLAP_CHARS,
    }
    if any(manifest.get(key) != value for key, value in expected_settings.items()):
        raise ValueError("Index was built with different model/retrieval settings. Rebuild with notebook 02.")
    if manifest.get("index_sha256") != _file_sha256(config.INDEX_PATH):
        raise ValueError("FAISS index checksum mismatch. Rebuild with notebook 02.")
    if manifest.get("chunks_sha256") != _file_sha256(config.CHUNKS_PATH):
        raise ValueError("Chunk table checksum mismatch. Rebuild with notebook 02.")
    if manifest.get("processed_kb_sha256") != config.sha256_file(config.KB_PATH):
        raise ValueError("Processed knowledge base changed since this index was built. Rebuild with notebooks 01 and 02.")

    index = faiss.read_index(str(config.INDEX_PATH))
    chunks = pd.read_json(config.CHUNKS_PATH, lines=True, dtype={"source": str, "title": str, "url": str, "text": str})
    _check_aligned(index, chunks)
    if manifest.get("num_chunks") != len(chunks) or manifest.get("vector_dimension") != index.d:
        raise ValueError("Index manifest does not match the loaded files. Rebuild with notebook 02.")
    return index, chunks
