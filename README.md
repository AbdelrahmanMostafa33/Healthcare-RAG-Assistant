# Healthcare RAG Assistant

A small retrieval-augmented generation (RAG) assistant that answers general health questions from a
curated knowledge base and shows the sources it used.

> **Education only.** This is a graduation project, not a medical device or a clinical service. It
> does not diagnose, does not recommend treatment, and has not been reviewed by clinicians.

## What it does

Given a general health question, the assistant either:

- answers from retrieved source passages, with `[n]` citations that resolve to the returned sources, or
- withholds the answer and says why (weak evidence, a time-sensitive question, a failed safety
  check, a citation that could not be validated, or an uncalibrated build), or
- replaces the answer with a fixed safety message for an emergency, a harmful request, or a request
  for personal medical advice.

The interesting part of the project is not the model call. It is the decision path around it: when
to answer, when to abstain, what "enough evidence" means, and how to measure that honestly.

## Architecture

```text
OFFLINE (once, in order)

  01_data_and_kb      MedQuAD, MedlinePlus XML, 3 USPSTF pages
                      -> clean, de-duplicate, one row per document
                      -> data/processed/kb_documents.csv
  02_build_index      -> chunks (~1000 chars, question line repeated on each chunk)
                      -> bge-m3 embeddings -> FAISS exact inner-product index
                      -> data/index/kb.faiss + chunks.jsonl + manifest.json
                      -> calibrate the evidence cutoff on dev questions only
                      -> reports/abstention_threshold.json

ONLINE (app.py, the notebooks and the evaluation all call this one pipeline)

  question
    | FastAPI validation (3-1000 characters)
    | src/safety.py           emergencies, harmful requests, personal-advice requests  -> fixed reply
    | src/safety_check.py     short Gemini classification for questions the rules do not match
    |                         (a failed check withholds the answer; it never falls through)
    | calibrated cutoff?      no -> withhold (fail closed)
    | freshness check         "latest / this week / right now" -> withhold
    | src/retriever.py        FAISS top 40 -> bge-reranker-v2-m3 -> up to 5 source pages
    | best rerank score < cutoff -> withhold (insufficient evidence), no model call
    | src/generator.py        Gemini, evidence-only prompt, citation instructions
    | citation check          every [n] must resolve to a supplied passage; one is required
    v
  answer + cited sources + disclaimer, or a fixed withholding message
```

Everything that matters is in `src/`, and there is exactly one implementation of each step: the API,
notebook 03 and the evaluation runner call the same `RAGPipeline` and the same helpers.

### Design decisions

| Decision | Why it is here | Evidence |
|---|---|---|
| Dense retrieval with `BAAI/bge-m3` | Strong multilingual encoder, but here it is simply a reasonable default for English health text | Not benchmarked against alternatives; a different encoder requires rebuilding the index **and** recalibrating |
| Cross-encoder rerank with `BAAI/bge-reranker-v2-m3` | Dense search returns topically similar chunks; the reranker orders them for the actual question, and its score drives the abstention decision | Not benchmarked against reranker-free retrieval, because there are no relevance labels to measure the difference fairly. Notebook 02 shows the scores it produces |
| Cutoff on the reranker score | One number that separates "the knowledge base covers this" from "it does not" | Calibrated on dev questions only (~0.9 target abstention recall). The value is a raw logit, **not** a probability |
| Two-stage safety | Local rules are predictable and testable; the Gemini check covers rewordings the rules miss | The local rules alone catch only a small share of a 50-question crisis/harm holdout (see below); this is measured, not assumed |
| Structural citation checking | Cheap, deterministic, and catches the failure that matters for trust: a source number that points at nothing | It does **not** prove that a passage supports a claim. See "Safety and limits" |
| Fixed replies instead of generated safety text | No provider call, no chance of a reassuring but wrong answer | Covered by tests |

Deliberately **not** in the project: agents, a database, a vector service, authentication,
deployment infrastructure, multiple provider frameworks. None of them is needed to demonstrate the
idea, and each would add a component to explain.

## Data

| Source | What is used | Licence / attribution |
|---|---|---|
| [MedQuAD](https://github.com/abachaa/MedQuAD) (`lavita/MedQuAD`, pinned revision in `src/config.py`) | Question/answer pairs from NIH sites (GARD, NIDDK, NIHSeniorHealth, ...) | CC BY 4.0 |
| [MedlinePlus](https://medlineplus.gov/xml.html) health-topic XML | One document per English topic, plus its "also called" aliases | See [MedlinePlus reuse guidance](https://medlineplus.gov/about/using/usingcontent/) |
| [USPSTF](https://www.uspreventiveservicestaskforce.org/) | Recommendation statements for breast, colorectal and blood-pressure screening | US federal government work |

Preprocessing (`src/kb.py`) is deliberately small and explainable:

- MedQuAD rows without an answer are dropped; duplicate questions keep the longer answer.
- MedQuAD's copy of the MedlinePlus health topics is dropped where the official XML already provides
  the same topic, so the same article cannot appear twice in one context.
- HTML is parsed (not regex-stripped); trailing navigation boilerplate is removed only when it looks
  trailing, so words like "More on:" inside an answer are preserved.
- Documents are split into ~1000-character chunks on sentence boundaries with a one-sentence overlap.
  Every chunk starts with the document's question, because it is the cheapest way to keep a chunk
  from the middle of a long answer interpretable — for an embedding and for a human reading it.
- `save_index` writes the FAISS index, the chunk table and a manifest together, and `load_index`
  refuses to load if the manifest, checksums, models or chunking settings disagree. A dataset change
  therefore cannot be served silently against an old index.

Generated data (`data/raw`, `data/processed`, `data/index`, `reports/`) is **not committed**. A fresh
clone has no knowledge base until you run the notebooks.

## Setup

Python 3.11 or 3.12. Building the knowledge base needs internet access and about 5 GB of disk. The
index itself is ~125 MB; most of the space is the two model checkpoints in the Hugging Face cache.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Then put a [Google AI Studio](https://aistudio.google.com/app/apikey) key in `.env` as
`GOOGLE_API_KEY`. That single key powers both answer generation and the safety check. A Groq or
Cerebras key is needed only for the optional LLM-judge evaluation. Never commit `.env`.

### Build the knowledge base and the index

Run the notebooks from the activated environment, in order:

1. `notebooks/01_data_and_kb.ipynb` — download, clean and inspect the sources; writes
   `data/processed/kb_documents.csv`.
2. `notebooks/02_build_index.ipynb` — chunk, embed, build and self-check the index; then calibrate the
   evidence cutoff on the dev questions and save `reports/abstention_threshold.json`.

Notebook 02 does not call any provider. On CPU, encoding ~32k chunks takes a while; a CUDA GPU makes
it much faster.

### Walk through one question

`notebooks/03_rag_pipeline.ipynb` follows a single question through search, reranking, the cutoff
decision and prompt construction, and then runs the same steps through `RAGPipeline` so you can see
that the notebook and the app agree. Live generation is switched off in the saved notebook.

### Run the web demo

```powershell
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000/>. The page reports its own state from `/health`:

- `loading` / `error` → HTTP 503
- `ok` (index present, cutoff calibrated) and `degraded` (index present, cutoff missing or stale) →
  HTTP 200. In `degraded` state the assistant withholds every answer rather than guessing.

`POST /query` has a 60-second budget. The deadline is passed into the pipeline, which checks it
between stages and caps each provider request's HTTP timeout at the remaining time, so a timed-out
request stops working instead of finishing in the background. A provider call already in progress
cannot be interrupted; only its socket timeout bounds it.

## Tests

```powershell
python -m pytest -q
```

The suite does not need an API key, a model download, a GPU or network access: it uses a small
in-memory FAISS index, fake encoders/rerankers and fake provider clients. It covers the safety rules,
the pipeline's decision path, citation parsing, deadline propagation, retry classification,
calibration, the metric definitions, the API contract and the data validators.

## Evaluation

`data/eval/dev_questions.csv` (168 questions) is the development set and the **only** source used to
choose the evidence cutoff. `data/eval/test_questions.csv` (87 questions) was reviewed during
development, so it is a second exploratory set, not an untouched holdout. There is no clinician-
reviewed or frozen holdout in this project, and the README does not claim one. The two files contain
no exact duplicate question, but ~30 pairs share a template ("What are the side effects of X?",
"I feel X. What is wrong with me?"), so test numbers say more about familiar phrasings than about a
fresh distribution.

Creating a genuinely untouched holdout, and having a clinician check a sample of the labels and
answers, is the most valuable next step for this project. It has not been done here.

Each question carries an `expected_behavior`: `answer`, `insufficient_evidence`, `emergency`,
`safety_redirect` or `refuse`. The evaluation compares that expectation with the outcome the pipeline
actually produced. The definitions live in `src/metrics.py`:

| Check | Definition | Denominator |
|---|---|---|
| Outcome accuracy | expected behaviour vs actual outcome | questions that produced a decisive outcome |
| Abstention recall | no-evidence questions withheld | no-evidence questions that reached an evidence decision |
| Abstention precision | withheld questions that should have been withheld | questions that were withheld for evidence/freshness reasons |
| Over-refusal | answerable questions withheld | answerable questions that reached an evidence decision |
| Citation validity | answers whose every `[n]` resolves to a supplied passage | answers that were returned |
| Judge pass rates (optional) | three LLM-judge checks: context relevance, faithfulness, answer relevance | questions the judge graded (excluding `not_applicable`) |

Three details that are easy to get wrong, and that this version handles explicitly:

1. **Safety replies are not evidence abstentions.** A redirect, a refused request, a failed safety
   check, a provider error and a citation rejection are reported as their own outcomes. They leave
   the abstention denominator instead of being scored as missed abstentions.
2. **The calibration figure and the evaluation figure are not the same quantity.** Calibration is a
   score-based estimate: is this question's best reranker score below the cutoff (or does the
   freshness policy withhold it)? It covers every dev question that reaches retrieval. At runtime a
   safety redirect or a failed safety check can replace that decision, so evaluation reports how many
   no-evidence questions were actually measurable and what happened to the rest. The calibration file
   records both numbers.
3. **A pass rate needs its denominator.** Judge scores are computed over the rows that were graded.
   100% on a small graded subset is not 100% system accuracy, and it says nothing about medical
   correctness.

Run the optional evaluation from `notebooks/04_evaluation.ipynb`. Provider-calling cells are off by
default and print the planned call count first. The deterministic run needs the generator key only;
set `USE_JUDGE = True` to add the LLM judge. Reports are written to
`reports/evaluation/<split>_answers.csv`, `<split>_metrics.json` and the judge CSVs. Notebook 04 only
displays a saved report when its evaluation version and cutoff match the current build.

**Reporting status:** this repository does not contain a verified end-to-end evaluation run for the
current code. The reports directory is generated locally and is not committed; run notebook 04 to
produce numbers for your own machine, and quote them together with the question count, the cutoff and
the model names. The judge is an unvalidated language model, so its scores are estimates.

## Safety and limits

The assistant is English-only. Local rules in `src/safety.py` run first, then a Gemini safety check
handles the rest. Every submitted question is checked; an unanswered-by-rules question is sent to
Gemini for that check, and a question that continues may also be sent to Gemini for generation.
Avoid entering names or other identifying details: those texts go to the provider.

Fixed safety messages are deliberately blunt, because a missed emergency is worse than an unnecessary
one. The local rules are keyword patterns, not a classifier.

**Regex-only development baseline** on the two frozen fixtures written for this project:

- 50-question holdout: urgent cases escalated were **2/24**
- 50-question holdout: harmful requests withheld were **1/10**
- 50-question holdout: educational questions left alone were **16/16**
- different-wording probes: urgent cases escalated were **6/42**
- different-wording probes: harmful requests withheld were **1/9**
- different-wording probes: educational questions left alone were **8/12**

These are small development fixtures, not evidence of clinical safety.

That is the honest baseline for the rules alone: they are reliable on the phrasings they were written
for and weak on new ones, which is why the pipeline also calls the Gemini safety check. The check's
own accuracy is **not** measured by this project; it is a second chance, not a guarantee.

Other limits worth stating plainly:

- **Citation checks are structural.** A valid `[n]` proves the passage was in the prompt and is
  returned as a source; it does not prove the passage supports the sentence. An answer containing any
  impossible reference (for example `[99]` next to a valid `[1]`) is withheld entirely rather than
  shown with a dangling source number.
- **Retrieval quality has no ground-truth labels here**, so no recall@k or precision@k is reported.
  Notebook 02 shows what retrieval returns for a few questions; the judge's context-relevance score
  is a fallback estimate, not a measurement.
- **The knowledge base is static and partially old.** MedQuAD republishes NIH pages that have since
  moved or been retired; some links are hidden in the UI and the app suggests searching for a current
  page instead. Questions that need current facts are withheld.
- **No clinical validation.** The question labels, the answers and the judge were all produced inside
  this project. None of it has been checked by a clinician, and none of it is a basis for real health
  decisions.

## Project files

```text
app.py                 FastAPI app: /, /query, /health
frontend/              Static demo page (no framework, no build step)
src/pipeline.py        The one orchestration path shared by app, notebooks and evaluation
src/safety.py          Local emergency / harmful / personal-advice rules and fixed messages
src/safety_check.py    Short Gemini safety classification for unmatched questions
src/retriever.py       FAISS search, reranking, source grouping, context selection
src/generator.py       Grounded generation, deadline-bounded retries, citation parsing
src/kb.py              Data cleaning, chunking, embeddings, index build/load
src/calibration.py     Dev-only cutoff selection and its metadata
src/metrics.py         Outcome, abstention, citation and judge metrics
src/evaluation.py      Evaluation runner and report writer
src/deadline.py        One absolute request deadline shared across stages
src/retry.py           Bounded retries for transient provider failures
src/config.py          Paths, model pins, provider settings, threshold validation
notebooks/             01 data+KB, 02 index+calibration, 03 walkthrough, 04 evaluation
data/eval/             Development and exploratory test questions (committed)
tests/                 Local tests and the safety fixtures (committed)
reports/               Generated calibration and evaluation output (not committed)
```

## Rebuilding after a change

| You changed | Rebuild |
|---|---|
| Source data or cleaning/chunking in `src/kb.py` | Notebooks 01 and 02, then recalibrate |
| Embedding or reranker model pin | Notebook 02 (index and cutoff) |
| `RAG_PIPELINE_VERSION` or anything that changes reranker scores | Notebook 02 calibration |
| Safety rules (`src/safety.py`) | Bump `SAFETY_RULES_VERSION`, then recalibrate |
| Prompt or generation settings | Re-run the evaluation; the cutoff does not depend on them |
| Evaluation questions | Notebook 02 calibration, then notebook 04 |

`config.load_threshold()` fails closed: a threshold whose fingerprint does not match the current
index, models, dev file, pipeline version or safety-rule version is ignored, the app reports
`degraded`, and every question is withheld until you recalibrate. That is a correctness safeguard, not
a formality — a cutoff measured against a different retriever is meaningless.

## Licence

Project code is MIT licensed; see `LICENSE`. The datasets and model weights keep their own licences;
see the table under "Data".
