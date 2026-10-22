"""Build and load the knowledge base: clean the sources, split them into chunks, embed, index."""
import hashlib
import html
import json
import logging
import os
import re
from urllib.parse import urlparse
from xml.etree import ElementTree

import faiss
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from src import config

logger = logging.getLogger(__name__)

# Navigation blocks are a trailing suffix of an NIH page, not a rule that fires anywhere in the text.
# "More on:" inside an answer introduces another content section ("... More on: Biology\nInfective
# Fasciola larvae (metacercariae) are found ..."), so cutting at the first occurrence deleted real
# text from 57 answers in the shipped MedQuAD snapshot.
BOILERPLATE = [
    "More on:", "Top of Page", "Related Pages", "Images and logos on this website",
    "This graphic notice means", "Go to:", "Back to Top",
]
# A marker in the last quarter of a document is part of the trailing navigation block even if a
# sentence follows it, which is how the copyright notice after "Top of Page" is still removed.
TRAILING_FRACTION = 0.75
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


def strip_boilerplate(text, removed_stats=None):
    """Drop a trailing navigation block, never text from the middle of an answer.

    A marker counts as boilerplate only when it looks trailing: either nothing after it is a
    sentence (just navigation labels such as "Top of Page Related Pages junk"), or it sits in the
    last quarter of the document. "Treatment works. Go to: the ER if pain persists." is untouched.
    When text is dropped, its length is appended to ``removed_stats`` so the notebook can report how
    much the rule actually removes.
    """
    if text is None or pd.isna(text):
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    cut = len(text)
    for marker in BOILERPLATE:
        index = text.find(marker)
        if index == -1:
            continue
        tail = text[index:]
        if index >= len(text) * TRAILING_FRACTION or not re.search(r"[.!?]", tail):
            cut = min(cut, index)
    if removed_stats is not None and cut < len(text):
        removed_stats.append(len(text) - cut)
    return text[:cut].strip()


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

    # Some old NIH links have verified current equivalents. Keep other original URLs internally so
    # retrieval can still recognize chunks from the same page; the API hides known retired links.
    title_key = df["title"].str.casefold().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    niddk = df["source"].eq("MedQuAD (NIDDK)")
    df.loc[niddk & title_key.str.startswith("am i at risk for type 2 diabetes"), "url"] = (
        "https://www.niddk.nih.gov/health-information/diabetes/overview/symptoms-causes"
    )

    df = df[(df["question"].str.len() > 5) & (df["text"] != "")]
    df = df[~df["question"].str.contains(r"\(are\)\s*\?", regex=True)]   # "What is (are) ?" = empty topic name
    # same question twice: keep the longer answer
    df = df.assign(length=df["text"].str.len()).sort_values("length", ascending=False)
    return df.drop_duplicates(subset="question").drop(columns="length").reset_index(drop=True)


def parse_medlineplus_xml(xml_bytes):
    """MedlinePlus topics XML -> title, URL, summary and search aliases (English only)."""
    rows = []
    for topic in ElementTree.fromstring(xml_bytes).findall("health-topic"):
        if topic.get("language") != "English":
            continue
        summary = topic.find("full-summary")
        aliases = [
            (node.text or "").strip()
            for tag in ("also-called", "see-reference")
            for node in topic.findall(tag)
            if (node.text or "").strip()
        ]
        rows.append({
            "title": topic.get("title", ""),
            "url": topic.get("url", ""),
            "summary": "".join(summary.itertext()).strip() if summary is not None else "",
            "aliases": list(dict.fromkeys(aliases)),
        })
    return pd.DataFrame(rows, columns=["title", "url", "summary", "aliases"])


def prepare_medlineplus(raw):
    """One KB row per health topic. MedlinePlus has no questions, so we use 'What is <topic>?'."""
    title = raw["title"].apply(clean_text)
    if "aliases" in raw.columns:
        aliases = raw["aliases"].apply(
            lambda values: [clean_text(value) for value in values]
            if isinstance(values, (list, tuple, set)) else []
        )
        aliases = aliases.apply(lambda values: list(dict.fromkeys(value for value in values if value)))
    else:
        aliases = pd.Series([[] for _ in range(len(raw))], index=raw.index)
    aliases = pd.Series(
        [[alias for alias in names if alias.casefold() != name.casefold()]
         for name, names in zip(title, aliases)],
        index=raw.index,
    )
    retrieval_question = pd.Series(
        [f"What is {name}?" + (f" Also called: {', '.join(names)}." if names else "")
         for name, names in zip(title, aliases)],
        index=raw.index,
    )
    df = pd.DataFrame({
        "source": "MedlinePlus",
        "title": title,
        "url": raw["url"].fillna("").astype(str),
        "question": retrieval_question.apply(clean_text),
        "text": raw["summary"].apply(html_to_text),
    })
    df = df[(df["text"] != "") & (df["title"] != "")]
    return df.drop_duplicates(subset="url").reset_index(drop=True)


def prepare_uspstf_pages(pages):
    """Prepare official screening recommendation pages as ordinary KB documents.

    Only the recommendation statement is retained, excluding site navigation and the much longer
    evidence review. ``pages`` contains ``title``, ``url`` and downloaded ``html`` values.
    """
    rows = []
    for page in pages:
        title = clean_text(page.get("title", ""))
        url = str(page.get("url", "")).strip()
        source_html = page.get("html", "")
        if not title or not url or not source_html:
            raise ValueError("Each USPSTF page needs a title, URL, and downloaded HTML")
        soup = BeautifulSoup(str(source_html), "html.parser")
        section = soup.select_one(".recommendation-statement")
        if section is None:
            raise ValueError(f"Could not find the recommendation statement in {url}")
        text = html_to_text(str(section))
        if not text:
            raise ValueError(f"The recommendation statement is empty in {url}")
        rows.append({
            "source": "USPSTF",
            "title": title,
            "url": url,
            "question": clean_text(f"What does the USPSTF recommend about {title}?"),
            "text": text,
        })
    return pd.DataFrame(rows, columns=["source", "title", "url", "question", "text"])


# MedQuAD republishes the MedlinePlus Health Topics, which this project also downloads as the
# official XML. Both copies then compete in retrieval for the same topic.
MEDQUAD_MEDLINEPLUS_SOURCE = "MPlusHealthTopics"


def medlineplus_slug(url) -> str:
    """Topic slug from either MedlinePlus URL form.

    MedQuAD links to the old host (nlm.nih.gov/medlineplus/<slug>.html); the official XML uses
    medlineplus.gov/<slug>.html. Only the slug is comparable.
    """
    path = urlparse(str(url)).path.rstrip("/")
    return re.sub(r"\.html$", "", path.split("/")[-1]).casefold()


def drop_medlineplus_duplicates(medquad, medlineplus):
    """Keep the official MedlinePlus XML copy of a topic and drop MedQuAD's duplicate.

    Measured on the shipped snapshot: 799 of 867 MedQuAD (MPlusHealthTopics) rows repeat a topic the
    XML already provides, which is why the same diabetes topic appeared twice in the top 5.
    """
    official = {medlineplus_slug(url) for url in medlineplus["url"]} - {""}
    duplicate = (
        medquad["source"].str.contains(MEDQUAD_MEDLINEPLUS_SOURCE, regex=False)
        & medquad["url"].map(medlineplus_slug).isin(official)
    )
    dropped = int(duplicate.sum())
    if dropped:
        logger.info("Dropped %d MedQuAD rows duplicated by the MedlinePlus XML", dropped)
    return medquad.loc[~duplicate].reset_index(drop=True)


def build_kb(medquad, medlineplus, additional_sources=None):
    """One KB table, with MedQuAD's MedlinePlus subset deduplicated against the XML."""
    documents = [drop_medlineplus_duplicates(medquad, medlineplus), medlineplus]
    if additional_sources is not None:
        missing = set(KB_COLUMNS[1:]) - set(additional_sources.columns)
        if missing:
            raise ValueError(f"Additional source rows are missing columns: {sorted(missing)}")
        documents.append(additional_sources[KB_COLUMNS[1:]])
    kb = pd.concat(documents, ignore_index=True)
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
            # Repeat one short sentence at chunk boundaries to keep nearby context together.
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

    encoder = SentenceTransformer(
        config.EMBEDDING_MODEL,
        revision=config.EMBEDDING_MODEL_REVISION,
    )
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
            "embedding_model_revision": config.EMBEDDING_MODEL_REVISION,
            "reranker_model": config.RERANKER_MODEL,
            "reranker_model_revision": config.RERANKER_MODEL_REVISION,
            "model_runtime_versions": config.model_runtime_versions(),
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
        "embedding_model_revision": config.EMBEDDING_MODEL_REVISION,
        "reranker_model": config.RERANKER_MODEL,
        "reranker_model_revision": config.RERANKER_MODEL_REVISION,
        "chunk_max_chars": config.CHUNK_MAX_CHARS,
        "chunk_overlap_chars": config.CHUNK_OVERLAP_CHARS,
    }
    if any(manifest.get(key) != value for key, value in expected_settings.items()):
        raise ValueError("Index was built with different model/retrieval settings. Rebuild with notebook 02.")
    # A pinned model with a newer library patch release usually still returns the same vectors, so
    # this is reported rather than rejected. Rebuild the index if retrieval scores actually move.
    if manifest.get("model_runtime_versions") != config.model_runtime_versions():
        logger.warning(
            "Index was built with different library versions (%s); rebuild it if retrieval scores shift.",
            manifest.get("model_runtime_versions"),
        )
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
