"""Build and load the knowledge base: clean the sources, split them into chunks, embed, index."""
import html
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
    if text is None or (isinstance(text, float) and pd.isna(text)):
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
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    plain = BeautifulSoup(str(text), "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", html.unescape(plain)).strip()


def prepare_medquad(raw):
    """lavita/MedQuAD rows -> one KB row per question/answer pair."""
    missing = MEDQUAD_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"MedQuAD data is missing columns {sorted(missing)}. Use the lavita/MedQuAD dataset.")

    # The GARD, A.D.A.M. and drug subsets ship without answers (copyright), so they drop out here.
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

def _split_long_sentence(sentence):
    """Run-on lists without full stops can be longer than a chunk, so split them on words."""
    parts, current = [], ""
    for word in sentence.split():
        if current and len(current) + 1 + len(word) > config.CHUNK_MAX_CHARS:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    return parts + [current]


def split_text(text):
    """Pack whole sentences into chunks of at most CHUNK_MAX_CHARS characters.
    The last sentence of a chunk is repeated at the start of the next one if it is short enough."""
    sentences = []
    for sentence in SENTENCE_END.split(text.strip()):
        if len(sentence) > config.CHUNK_MAX_CHARS:
            sentences.extend(_split_long_sentence(sentence))
        elif sentence:
            sentences.append(sentence)

    chunks, current = [], []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence])) > config.CHUNK_MAX_CHARS:
            chunks.append(" ".join(current))
            last = current[-1]
            fits = len(last) <= config.CHUNK_OVERLAP_CHARS and len(last) + 1 + len(sentence) <= config.CHUNK_MAX_CHARS
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
        for body in split_text(doc.text):
            if len(body) < MIN_CHUNK_CHARS:
                continue
            text = f"{doc.question}\n{body}" if with_question and doc.question else body
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
    index = faiss.IndexFlatIP(embeddings.shape[1])   # exact search; fast enough at this size
    index.add(embeddings)
    return index


def _check_aligned(index, chunks):
    if index.ntotal != len(chunks):
        raise ValueError(f"Index has {index.ntotal} vectors but the chunk table has {len(chunks)} rows. Rebuild with notebook 02.")
    if not (chunks["chunk_id"].to_numpy() == np.arange(len(chunks))).all():
        raise ValueError("chunk_id must equal the row position in the chunk table.")


def save_index(index, chunks):
    _check_aligned(index, chunks)
    config.INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_PATH))
    chunks.to_json(config.CHUNKS_PATH, orient="records", lines=True, force_ascii=False)


def load_index():
    if not (config.INDEX_PATH.exists() and config.CHUNKS_PATH.exists()):
        raise FileNotFoundError("Index files not found. Run notebooks 01 and 02 to build them.")
    index = faiss.read_index(str(config.INDEX_PATH))
    chunks = pd.read_json(config.CHUNKS_PATH, lines=True, dtype={"source": str, "title": str, "url": str, "text": str})
    _check_aligned(index, chunks)
    return index, chunks
