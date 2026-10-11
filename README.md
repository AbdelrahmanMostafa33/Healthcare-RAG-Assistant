# Healthcare RAG Assistant

**An evidence-grounded RAG system for answering general health questions from curated healthcare sources.**

Healthcare RAG Assistant is an educational health-information assistant. It answers general questions about diseases, symptoms, causes, prevention, tests and treatments from curated NIH sources, shows which sources it used, and **abstains when the knowledge base does not contain enough relevant information**.

> **Not medical advice.** It is an information assistant, not an AI doctor. It does not diagnose, give personal advice, recommend doses, or replace a doctor or pharmacist.

## Problem

Health information online is large, scattered and uneven. Search engines return pages, not answers. A plain LLM gives fluent answers but cannot show where a claim comes from, and it may state wrong things confidently. For health questions, an answer you cannot trace to a source is hard to trust.

RAG (retrieval-augmented generation) addresses this: the system first searches a curated knowledge base, then asks the LLM to answer **only** from the passages it found, citing them. If nothing relevant is found, it does not ask the LLM at all.

## Scope

| | |
|---|---|
| **In scope** | Diseases and conditions: what they are, their symptoms, causes and risk factors, prevention, tests and diagnosis, and how they are treated in general |
| **Out of scope** | Specific medicines (doses, side effects, interactions, whether to start, stop or combine), diagnosing the user, personal advice, interpreting someone's test results, and live or current information |

**Why medicines are out of scope.** The sources describe how conditions are treated in general, not the uses, doses, side effects, or interactions of specific drugs. MedQuAD's drug rows and MedlinePlus drug pages are not part of the indexed corpus. Questions about named drugs therefore lack dedicated evidence and should abstain; the evaluation includes these questions to check that boundary.

An assistant that abstains on medicines is safer than one that answers from its own memory: every abstention is an explicit "I don't know", not a confident guess.

## How it works

```
User question
  -> safety rules (src/safety.py), two small keyword checks
        emergency        -> fixed emergency message with emergency numbers   (no retrieval, no LLM)
        personal advice  -> fixed redirect to a doctor or pharmacist         (no retrieval, no LLM)
  -> embed the question                                    (BAAI/bge-m3)
  -> FAISS: the 20 closest chunks                          (exact inner-product search)
  -> cross-encoder rerank, keep the best 5                 (BAAI/bge-reranker-v2-m3)
  -> check evidence: best rerank score >= threshold ?
        no  -> abstain (no LLM call)
        yes -> grounded prompt with numbered passages -> one LLM call   (gpt-oss-120b on Groq)
  -> answer with [1], [2] citations + the sources used + disclaimer
```

| Step | Why it is there |
|---|---|
| Safety rules | An acute emergency or a request for a personal dose or diagnosis should never get a general-information answer. The rules only match when the question describes a real person ("I have...", "my son...") and are tested against educational questions they must not catch ("What are the symptoms of a heart attack?", "What causes colic in babies?"). They are keyword rules, so they miss rephrasings; they are a safety net, not triage. |
| Dense retrieval | Finds chunks about the same thing as the question even when the wording differs. Fast. |
| Rerank | The cross-encoder reads the question and a chunk together, so it judges relevance better than comparing two separate embeddings. Too slow for the whole index, so it only sees the 20 candidates. Its score is also the evidence signal for the next step. |
| Evidence check | Retrieval always returns something. Without a check, the LLM would be asked to answer from irrelevant text. The threshold is chosen on the dev questions (see Evaluation). |
| Grounded prompt | Numbered passages and strict rules (use only these, cite them, say what is missing, no personal advice) make answers checkable. |
| Sources | Only the passages that were in the prompt are returned, with source, title and URL. |

There is no query classifier, no second retrieval pass, no retry with a looser prompt and no fallback to the model's own knowledge.

## Knowledge base

| Source | Content |
|---|---|
| [MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD) | Question/answer pairs written by NIH websites (NIDDK, CDC, NHLBI, ...). Kept with each document's URL and publisher so answers can be cited. |
| [MedlinePlus](https://medlineplus.gov/xml.html) | One plain-language summary per health topic (NLM). |

Documents are cleaned (HTML and NIH site text removed, duplicates dropped) and split into sentence-aware chunks of at most 1000 characters, with long run-on sentences split at word boundaries. A short boundary sentence may overlap between chunks (at most 200 characters). The document's question is placed on top of every chunk so it keeps its topic; space is reserved for that header before splitting, so the total embedded text still respects the cap. Chunk size and overlap are sensible defaults, not benchmark-tuned values.

**Why a small knowledge base.** The sources were chosen to match the questions, not to be large. PubMedQA (research abstracts with generated questions) and MedRedQA (informal Reddit replies) were left out because they fit consumer questions poorly. This is a judgement about fit; their effect on retrieval was not measured.

**The medication gap.** In the checked `lavita/MedQuAD` snapshot, drug, A.D.A.M., and herbal-supplement rows have no answer text and are filtered during ingestion. GARD is not wholly omitted: answer-bearing rare-disease rows are retained. MedlinePlus drug pages contain AHFS Consumer Medication Information, so they are not ingested. FDA labels (DailyMed / openFDA) are a possible future source that would need a separate ingestion step. Notebook 01 shows the answer availability by source; notebook 04 reports medication questions separately.

The index is not stored in the repository. Notebooks 01 and 02 build it.

## Tech stack

Python, pandas, BeautifulSoup, sentence-transformers (embeddings and reranker), FAISS, the Groq API through the OpenAI SDK, FastAPI with Pydantic, pytest, Jupyter.

## Quickstart

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt  # with a GPU, install the matching torch build first
cp .env.example .env                      # then add your GROQ_API_KEY
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

1. Build the knowledge base and index by running `notebooks/01_data_and_kb.ipynb`, then `notebooks/02_build_index.ipynb`. A GPU is recommended for the embedding step.
2. Calibrate the evidence threshold on the **dev** questions. This downloads/loads the retrieval models but does not call Groq and does not need an API key:

```bash
python -m scripts.calibrate_threshold
```

   Calibration is tied to the configured embedding/reranker model IDs, the evaluation CSV, and the exact index manifest. If any of these change, the saved threshold is rejected and the assistant abstains until recalibrated. The upstream model IDs are not pinned to immutable Hub commits, so check for model updates before treating a saved calibration as reproducible.
3. Add your Groq key to `.env`, then start the API and ask a question:

```bash
uvicorn app:app
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the symptoms of type 2 diabetes?"}'
```

Interactive docs are at `http://localhost:8000/docs`. Startup loads the models, so it takes a little while. It fails immediately, with a clear message, if `GROQ_API_KEY` is missing or the index has not been built.

4. Run the tests. They use small fakes, so no model weights, GPU, index files or API key are needed (install `requirements.txt`, including `faiss-cpu`, first):

```bash
pytest
```

To run the LLM-graded evaluation, open `notebooks/04_evaluation.ipynb` with `jupyter lab notebooks/04_evaluation.ipynb` after configuring `GROQ_API_KEY`. The notebook runs the dev evaluation by default; its LLM-only baseline is off to avoid extra provider calls. After freezing code and the dev-calibrated threshold, set `RUN_TEST = True` in the setup cell and rerun the notebook from the top for the held-out test split.

## API

`POST /query` with `{"question": "..."}` (3 to 1000 characters) returns:

| Field | Meaning |
|---|---|
| `answer` | The answer with `[n]` citations, the abstention message, or a fixed emergency or redirect message |
| `abstained` | `true` when the sources did not contain enough relevant evidence |
| `emergency` | `true` when the question described an emergency (fixed message, no retrieval, no LLM) |
| `redirected` | `true` when the question asked for personal medical advice (fixed redirect to a doctor, no retrieval, no LLM) |
| `sources` | Only passages cited in the answer: `n` (matches the citation), `source`, `title`, `url`. Empty unless an answer was generated. |
| `disclaimer` | The medical disclaimer |

`GET /health` returns the status, threshold and calibration state. An uncalibrated service reports `degraded`; normal questions are safely abstained until a valid calibration exists. A threshold is rejected if its embedding model, reranker, evaluation dataset, or index manifest does not match the current build.

## Notebooks

Read them in order and run each notebook from top to bottom. All notebooks are shipped without saved cell outputs so that stale model results are not mistaken for results from the current code. Notebook 01 downloads source snapshots; notebook 02 builds the index; notebook 04 performs calibration and LLM-based evaluation.

| Notebook | What it shows |
|---|---|
| `01_data_and_kb` | Scope, sources and why, cleaning, the final knowledge base, and a check of how well it covers the scope |
| `02_build_index` | Chunking, embeddings, the FAISS index, and checks that the index is correct |
| `03_rag_pipeline` | One question walked through retrieval, rerank, evidence check, prompt and LLM call by hand. Then the same thing through `RAGPipeline`, with an emergency, a personal-advice question and an abstention |
| `04_evaluation` | Choosing the abstention threshold, optional LLM-only baseline, RAG vs the baseline, failures, citation-reference checks, and `reports/results.md` |

## Evaluation

**Questions.** `data/eval/questions.csv` has 255 questions, split into dev (168) and test (87). Answerable questions have key facts; the others have an expected behaviour (`insufficient_evidence`, `emergency`, `safety_redirect` or `refuse`).

**Metrics** (all computed by notebook 04, graded by a judge model that is not the generator, every verdict saved in `reports/`):

| Layer | Metrics |
|---|---|
| Generation | Key-fact coverage on questions that should be answered |
| Behaviour | Pass rate for answer, emergency, refuse, safety-redirect and insufficient-evidence questions; abstention recall and precision; over-refusal; latency |

**Abstention threshold.** Chosen on the dev split as the lowest threshold that still abstains on at least 90% of the `insufficient_evidence` questions that reach retrieval. This maximizes answerable coverage subject to that safety floor. Run `python -m scripts.calibrate_threshold` after building the index; it uses no generator calls. Notebook 04 uses the same helper for its analysis. The test split should be run once with the resulting threshold frozen. The uploaded project archive intentionally does not include a calibrated threshold or generated reports: those artifacts are specific to an exact data/index/model build and must be recreated locally.

### Results

Evaluation outputs are generated by `notebooks/04_evaluation.ipynb` and saved under `reports/`. The uploaded archive has no generated evaluation report, so there is no current run result to claim here. Earlier exploratory runs were not reproduced after the latest safety, calibration, and citation-validation changes; do not treat older numbers as current performance.

For interpretation, compare systems on the same questions and labels, report dev and frozen test separately, and review failures—not just a single average:

The LLM-only baseline may score highly on common health questions even when it does not provide verifiable sources. RAG's intended advantages are source-grounded answers, visible citations, and explicit abstention when evidence is weak; these are hypotheses to test, not guaranteed performance improvements. The current code includes citation-reference validation and a fail-closed pre-calibration state, but the final safety and answer-quality rates must be measured on a fresh run.

## Examples

**Abstention.** Deterministic; no LLM call is made:

```json
{
  "answer": "I couldn't find enough relevant information in my current sources to answer this reliably. Try rephrasing the question, or ask a doctor or pharmacist. If you think this could be an emergency, call your local emergency number or go to the nearest emergency department now.",
  "abstained": true,
  "emergency": false,
  "redirected": false,
  "sources": [],
  "disclaimer": "This is general health information for education only. ..."
}
```

**Emergency and personal advice.** "I have crushing chest pain that spreads to my left arm" returns the fixed emergency message with `"emergency": true`. "How much ibuprofen can I give my 4 year old?" returns a fixed redirect to a doctor or pharmacist with `"redirected": true`. Neither calls retrieval or the LLM. "What are the symptoms of measles in children?" is an educational question and goes through retrieval as normal.

**Grounded answer: the response shape.** The text and sources depend on the knowledge base and the model. Notebook 03 prints real outputs.

```json
{
  "answer": "<answer using only the passages, with [1] [2] citations>",
  "abstained": false,
  "emergency": false,
  "redirected": false,
  "sources": [
    {"n": 1, "source": "MedlinePlus", "title": "<topic>", "url": "https://medlineplus.gov/..."},
    {"n": 2, "source": "MedQuAD (<publisher>)", "title": "<topic>", "url": "https://..."}
  ],
  "disclaimer": "..."
}
```

## Limitations

- **Educational use only.** No diagnosis, no personal advice, no doses. It knows nothing about the user.
- **Safety rules are keywords.** They catch the common phrasings and miss others ("I can't feel my left arm and my chest hurts" is caught; many rewordings will not be). A missed emergency falls through to retrieval, where the prompt still tells the model to point to emergency services. The emergency numbers and crisis lines in `src/safety.py` include Egypt, the US, the EU and the UK. For Egypt, the Ministry of Health lists mental-health support at **16328**, **08008880700**, and **0220816831** in its [hotline notice](https://www.mohp.gov.eg/NewsDetails.aspx?subject_id=5901).
- **Knowledge base.** NIH consumer-health content in English. It is a static snapshot: MedQuAD is 2019 data, so some content is dated, and there is no live information. There is no drug source.
- **Retrieval limits the answer.** If the right passage is not retrieved the answer is incomplete, and if the threshold is too strict, answerable questions are refused. No gold relevant-passage labels are currently provided, so recall@k and nDCG are not claimed; a small clinician/reviewer-labeled retrieval set would be the next meaningful evaluation upgrade.
- **Evaluation.** Key facts were written by the project owner and have not been reviewed by a clinician. Some answer behavior is graded by an LLM judge (the default judge model is a provider preview model). The threshold is tuned on the dev split, which is a noisy estimate. The safety rules were written while looking at some evaluation questions, so measured safety rates can be optimistic. Citation-reference checks verify that citation IDs exist; they do not prove every claim is entailed by its cited source.
- **Not reproduced end to end in this archive.** The index is generated, not shipped. Encoder/chunk-size alternatives were not compared, and performance claims should wait for a fresh dev calibration and frozen test evaluation after the current changes.

## What I tried and removed

An earlier version was reviewed and simplified. The evidence behind each decision differs, so it is stated:

| Removed | Reason | Evidence |
|---|---|---|
| Query classifier and category routing | Trained on labels from a keyword rule that agreed with hand labels on only about 55% of the 150 evaluation questions. In retrieval its categories were compared with chunk categories using a different vocabulary, so it could not change which chunks came back. It also added a model download and a hard dependency. | Label agreement measured; routing mismatch found by code review |
| BM25 / hybrid retrieval | Switched off by default while the documentation described hybrid search. Nothing in the repository showed a gain. It should come back only if it beats dense + rerank in notebook 04. | No measurement either way |
| Fallbacks that answer without evidence (the model's own knowledge, a looser retry prompt, returning a raw chunk) | They contradict the goal of answering only from sources. Replaced by abstention. | Design decision |
| PubMedQA and MedRedQA in the knowledge base | Poor fit for consumer questions, and they dominated the index by size. | Judgement, not measured |
| Chunks made of the question and nothing else | The earlier chunking produced 219,458 of them (23.2% of all chunks). Now the text is split first and the question added afterwards. | Measured |
| API key rotation, API auth, response cache, extra middleware, a static dashboard | Not useful for this project. Key rotation also made the tests need a real API key even with a fake client. | Review; 11 tests failed without a key |
| Gemini judge (`google-genai`) | Its only recorded run failed ("client has been closed": a throwaway `Client` object is garbage-collected, which closes its connection). The saved results were judged on Groq. One judge, one client. | Run error; library source |
| Medication keyword list and the broad "children / should I / is it safe" redirect rules | They sent educational questions ("symptoms of measles in children", "How can I lower my blood sugar?") to the emergency page or the redirect. Replaced by narrow rules in `src/safety.py`; named drugs now go through retrieval and are abstained on by the threshold. | On the 31 educational questions in `tests/test_safety.py`: 23 intercepted by the old rules, 0 by the new ones (a regression test written partly from the failures found, not an unbiased accuracy estimate) |
| BLEU / ROUGE / BERTScore | Not meaningful for checking facts and groundedness. | Judgement |

## Project structure

See [`AUDIT.md`](AUDIT.md) for the engineering decisions, repair summary, verification limits, and recommended rebuild/evaluation sequence.

```
AUDIT.md                 architecture, changes, data/evaluation caveats and verification boundary
app.py                  FastAPI service (POST /query, GET /health)
src/
  config.py             settings and calibration/index compatibility checks
  kb.py                 clean sources, chunk, embed, build and validate the FAISS index manifest
  calibration.py        dev-only threshold scoring and safe artifact saving
  retriever.py          embed -> FAISS top 20 -> rerank -> top 5
  safety.py             emergency and personal-advice rules that run before retrieval
  generator.py          grounded prompt and the single LLM call
  retry.py              bounded provider rate-limit retries shared by generation and evaluation
  pipeline.py           RAGPipeline.answer(): safety -> retrieve -> evidence check -> generate -> sources
  evaluation.py         judge, scoring, threshold choice, abstention metrics, report tables
notebooks/              01 data, 02 index, 03 pipeline, 04 evaluation
data/eval/README.md     evaluation labels, scope and limitations
scripts/calibrate_threshold.py  calibrate the threshold without a Groq API key
data/eval/questions.csv the 255 development/test questions
reports/                local calibration and evaluation artifacts (not included in this archive)
tests/                  behaviour tests (no model downloads, index, or API key needed)
```

## License

MIT, see `LICENSE`.
