# Healthcare RAG Assistant

A source-grounded health information assistant that demonstrates practical AI engineering: data ingestion, vector retrieval, evidence-based abstention, citation validation, grounding checks, safety rules, and honest evaluation. This is an educational portfolio project, not a clinical system. It does not diagnose, prescribe, or replace professional care.

## Architecture

```text
question → safety rules → emergency help, personal-advice redirect, or harmful-request refusal
  → embed and search the cleaned MedQuAD + MedlinePlus corpus with FAISS (top 20)
  → BGE reranker (top 5) → calibrated evidence threshold
  → weak evidence: abstain | sufficient evidence: grounded LLM answer
  → validate citation IDs → answer with cited sources
```

The FastAPI service and notebooks use the same implementation in `src/`. Emergency, high-risk personal-advice, and explicit harmful-request checks run before retrieval. The pipeline abstains until a threshold has been calibrated for the current evaluation data, index, and safety rules. Generated answers must cite retrieved passages; structurally invalid citations cause a separate abstention rather than counting against the evidence threshold. The safety module is a single shared source: the notebooks, the evaluation harness, and the FastAPI service all import the same `src/safety.py`, so a safety fix in that file is reflected everywhere at once. A saved threshold records `SAFETY_RULES_VERSION` from that file, and a version mismatch makes the pipeline abstain until it is recalibrated, so a deployment never runs on a threshold computed against different rules than the live code. Versioning rather than hashing the file means a comment, a docstring edit or a CRLF checkout cannot silently invalidate a calibration; bump the constant when a pattern or constant actually changes. The other fingerprint files (`questions.csv`, the index manifest) are hashed with line endings normalized for the same reason.

- Exact FAISS inner-product search over normalized vectors keeps retrieval simple for this corpus.
- Retrieval uses [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) embeddings and [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) cross-encoder scores. Embedding, reranker, and MedQuAD snapshots are pinned to immutable revisions. The index manifest records model revisions, score-sensitive library versions, and file hashes. An index built with a different model, revision or chunk setting is rejected, as is a calibration whose evaluation data or safety-rules version has changed; a difference only in the library version numbers is reported as a warning rather than refusing to serve, since the model snapshots are still pinned and rebuilding an index or recalibrating is the expensive response.
- **Generation and judging use separate providers.** Gemini (`GENERATOR_MODEL`, default `gemini-3.5-flash-lite`) generates answers; Groq's GPT-OSS 120B judges them. Generation runs at temperature 0 so an evaluation is reproducible against a frozen threshold (`GENERATOR_TEMPERATURE` overrides it for interactive demos). Quotas are project/organization scoped, so this separates the workloads but does not remove provider limits. Each API key variable accepts comma-separated keys: on HTTP 401, that key is skipped; on HTTP 429 or an explicit quota rejection, it is cooled down and the next key is tried once, with the retry budget drained so a retried key is not re-entered. Google applies Gemini rate limits per project rather than per API key, so keys from one project share a single RPM/TPM/RPD budget and rotation may not add capacity. Credentials are never written to logs.
- Citation checks verify that IDs refer to supplied passages. They do not prove semantic support for every claim.

## Design choices

I reviewed the [OpenAI Knowledge Retrieval starter](https://github.com/openai/openai-knowledge-retrieval), Haystack's [RAG evaluation example](https://github.com/deepset-ai/haystack/blob/main/e2e/pipelines/test_evaluation_pipeline.py), and the Sentence Transformers [retrieve-and-rerank example](https://github.com/huggingface/sentence-transformers/tree/main/examples/sentence_transformer/applications/retrieve_rerank). The useful choices here are inspectable pipeline stages, a two-stage retriever for a corpus too large for a cross-encoder scan, and explicit failure-case evaluation. This project keeps one shared Python pipeline and a small local evaluation harness instead of adding a framework, hosted vector store, or separate frontend. Retrieval recall is left unreported until relevant passages are labeled.

## Data

- **MedQuAD**: downloads the `lavita/MedQuAD` Hugging Face dataset at the revision in `src/config.py`. The original [MedQuAD project](https://github.com/abachaa/MedQuAD) describes the collection and publishes a [CC BY 4.0 license](https://github.com/abachaa/MedQuAD/blob/master/LICENSE.txt); rows without answer text are excluded and source publishers are retained. MedQuAD republishes the MedlinePlus Health Topics under the `MPlusHealthTopics` publisher, which the official XML below also provides; the duplicated MedQuAD rows are dropped by URL slug so one topic cannot occupy two of the five retrieved passages. The corpus is still skewed towards rare genetic disease: GHR and GARD together are about two thirds of the rows.
- **MedlinePlus**: English health-topic summaries from the official [XML downloads](https://medlineplus.gov/xml.html). MedlinePlus identifies these summaries as public-domain content and requests source attribution; see its [reuse guidance](https://medlineplus.gov/about/using/usingcontent/). The download is a dated snapshot rather than an immutable source revision; the manifest records its URL and checksum. Other MedlinePlus material can have separate copyright terms and is not included in this project's stated data license.
- **Evaluation**: 255 hand-curated questions in `data/eval/questions.csv` (168 dev, 87 test). The project owner authored labels and key facts; clinicians have not reviewed them. Related topics can occur across splits, so the test set is only a small held-out check. There are no labeled relevant passages; retrieval recall@k and nDCG are not reported. `data/eval/safety_probes.csv` holds 88 further questions that are used only to measure the pre-retrieval safety rules; `tests/test_safety_probes.py` requires every live-emergency probe to be caught and every educational probe to reach retrieval.

Downloaded sources, the processed knowledge base, index, calibration, and evaluation outputs are generated locally and not committed, and notebook outputs are stripped before committing: run the notebooks to see their numbers. See `data/eval/README.md` for label definitions and caveats.

## Setup

Use Python 3.11 or 3.12. Model weights require several GB of disk; a GPU is optional. Initial data/model downloads need network access. Generation needs a Google AI Studio API key; LLM-judged evaluation needs a Groq API key. Both use provider quota.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set GOOGLE_API_KEY and GROQ_API_KEY to authorized keys.
```

### Linux or macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY and GROQ_API_KEY to authorized keys.
```

For a compatible NVIDIA GPU, install the PyTorch build matching your CUDA version before the remaining requirements. CPU use is supported but slower.

## Build and run

From the repository root:

1. `jupyter lab notebooks/01_data_and_kb.ipynb` downloads source data, records provenance metadata, and builds the cleaned knowledge base.
2. `jupyter lab notebooks/02_build_index.ipynb` downloads the pinned embedding/reranker models and builds the FAISS index. Repeat after changing data, chunk settings, or model revisions.
3. Calibrate the evidence threshold on dev only in the first section of the evaluation notebook:

   ```powershell
   jupyter lab notebooks/04_evaluation.ipynb
   ```

   Run the setup cell, the pre-flight cell and section 1, then stop if you only need calibration. The pre-flight cell loads no models and calls no provider: it reports the prerequisite files, the saved threshold, the index manifest and the configured credential counts, so a run that would silently abstain on every question is caught before any API call. Section 1 loads the retrieval models and saves the threshold, also without provider API keys or generation calls.

4. The optional pipeline walkthrough is `jupyter lab notebooks/03_rag_pipeline.ipynb`. Live generation is disabled by default; opt in for one answer-generation step when the evidence threshold allows it. Each request tries configured keys at most once when authentication or quota errors occur. Transient 429 and 5xx responses (including the client timeout the Gemini SDK reports as 504) can trigger up to five bounded retries, a provider-requested delay is honored up to 60 seconds so that one request cannot hold a worker thread for an hour, and the retry budget is drained on a retried key so it is not re-entered later. Each generation call waits up to `GENERATOR_TIMEOUT_MS` (default 240 s), a ceiling for a busy provider rather than a target: the default `gemini-3.5-flash-lite` answers a five-passage prompt in about a second, while a slower model chosen through `GENERATOR_MODEL` (`gemini-3.8-flash` was measured at 58-131 s per call) makes that ceiling necessary, because an earlier 60 s client timeout turned ordinary queuing into a 504 that looked like a provider failure. Emergency, personal-advice, and explicit harmful-request examples do not call the model.
5. Start the API after the knowledge base, index, and matching calibration exist:

   ```powershell
   uvicorn app:app --reload
   ```

   Open `http://127.0.0.1:8000/docs`, or query from PowerShell:

   ```powershell
   Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/query `
     -ContentType 'application/json' `
     -Body '{"question":"What are common symptoms of type 2 diabetes?"}'
   ```

`GET /health` reports readiness and calibration state. The first startup loads local model weights and can take time. Missing credentials or index files produce a clear startup error.

## Evaluation

Run `jupyter lab notebooks/04_evaluation.ipynb`. Its setup and pre-flight cells need no credentials: the pre-flight cell reports the prerequisite files, the saved threshold, the index manifest and the configured credential counts, and loads no models. Its first section then calibrates on dev retrieval scores without API keys; to calibrate only, stop after section 1. Continuing initializes Gemini and Groq clients, runs the dev RAG evaluation, and requires their API keys. Gemini generates the answers, and Groq GPT-OSS 120B grades them with a separate judge prompt. `RUN_LLM_BASELINE` is off by default because it makes additional API calls. Set `RUN_TEST = True` only after code and the dev threshold are frozen; then run the notebook top to bottom once for the held-out split.

The notebook saves per-question answers, fact grades, behavior grades, faithfulness verdicts, key-fact-in-context grades, judge coverage, citation-reference checks, and a summary under `reports/`. Answers are saved before LLM grading begins. One judge grading operation covers key facts and expected behaviour for each answer. Malformed JSON is marked as a grading error instead of triggering another judge generation; transient 429s and 5xx replies still use bounded retries. A row that fails for any other reason - a malformed grade, a transient outage, a client timeout - is recorded with its cause and the run continues, so one bad row cannot discard every grade after it. Only hard quota exhaustion or an exhausted rate-limit budget stops a run; completed work is then saved and remaining questions are marked skipped. Failed, skipped, and ungraded rows are reported separately and excluded from answer-quality denominators; check coverage before interpreting metrics. The LLM-only comparison is shown only when both systems complete generation and judging for the full same split.

Measures include key-fact coverage (LLM-judged), faithfulness (supported claims over total claims, judged against the retrieved passages with no reference answer - the [RAGAS](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) definition), a key-fact-in-context retrieval proxy that reuses the existing key facts as labels, expected-behavior pass rates, abstention recall/precision, over-refusal, abstention on scope/policy questions, 95% Wilson intervals for the abstention rates, citation-ID validity, latency, and run coverage/failures. Faithfulness and key-fact-in-context are reported separately from key-fact coverage and abstention, because they answer different questions. The faithfulness judge is another LLM call per graded answer, so it carries provider cost and the same proxy caveats as the other judge metrics. Abstention counts the threshold decision only: an answer withheld because its citations failed validation is reported separately as `citation_validation_failed`, and a question answered by a safety rule never reaches the threshold at all. The threshold is selected on dev to meet a minimum abstention-recall target for the questions the corpus genuinely has no evidence for (types `unanswerable` and `live_info`). Scope and policy questions - drug questions the corpus deliberately excludes, vague personal symptoms, out-of-scope research - score in the same range as answerable ones, so they are reported as their own slice instead of raising the threshold and refusing answerable questions. This is a small, noisy calibration set, not a safety guarantee. The judge is an imperfect proxy, labels have not been clinician-reviewed, and safety rules were developed while inspecting this set. Gemini and Groq model IDs and behavior can change on the provider side; reports record IDs, but they are not immutable model snapshots. Do not infer clinical performance or causal improvement from these metrics. No metrics are prefilled here.

## Tests

The unit suite uses fake clients and small local data. It needs no model weights, GPU, generated index, or API credentials.

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

There is a single `requirements.txt`: the suite imports `faiss`, `numpy`, `pandas`, `fastapi`, `pydantic`, and the provider SDKs, so a smaller test-only list would not be able to collect it. GitHub Actions runs the same suite on pushes and pull requests against `requirements.txt`.

## API and limitations

- `POST /query` accepts `{"question": "..."}` and returns an answer, emergency/redirect/refusal/abstention flags, sources actually cited, and a general-information disclaimer.
- `GET /health` reports readiness and whether calibration matches the current data, index, models, and `src/safety.py` bytes (the threshold file is rejected when its stored safety hash does not match).

The synchronous API is intended for a local demo, not production traffic. Safety checks are narrow keyword rules and can miss rephrasings; they normalize phone-typed curly apostrophes to ASCII before matching, so the crisis contacts and rules are consistent across a phone keyboard, the notebook evaluation harness, and the API. The fixed crisis contacts are limited to selected regions plus an international directory. The English NIH consumer-health corpus may be incomplete or dated. The system can abstain when useful content is missing and can still produce unsupported claims despite structural citation checks. Do not submit personal or sensitive health information.

The crisis message currently lists Egypt's Ministry of Health line (**16328**), the US **988** Lifeline, the UK Samaritans line (**116 123**), and [Find A Helpline](https://findahelpline.com). It normalizes phone-typed curly apostrophes to ASCII before matching, so text typed on a phone reaches the same safety result as typed text. Check the current information from the [Egyptian Ministry of Health](https://www.mohp.gov.eg/NewsDetails.aspx?subject_id=5901), [988 Lifeline](https://988lifeline.org/contact-us/), and [Samaritans](https://www.samaritans.org/how-we-can-help/contact-samaritan/) before relying on it. Generation sends the question and retrieved passages to Google Gemini; evaluation sends generated answers and grading criteria to Groq.

## Repository map

```text
app.py                         FastAPI endpoints
src/config.py                  paths, model snapshots, calibration validation
src/kb.py                      cleaning, dedup, chunking, embeddings, FAISS index + manifest
src/retriever.py               dense retrieval and cross-encoder reranking
src/safety.py                  emergency, harmful-request and personal-advice rules
src/generator.py               Gemini grounded prompt, generation, citation validation
src/provider_clients.py        Gemini generator and Groq judge clients
src/retry.py                   bounded retries for transient rate limits and 5xx failures
src/pipeline.py                shared safety/retrieval/abstention/generation flow
src/calibration.py             dev-only threshold calibration
src/evaluation.py              batching, scoring, judge, and report helpers
notebooks/                     data, index, pipeline walkthrough, evaluation
data/eval/                     hand-authored labels, splits and safety probes
tests/                         local unit, API and safety-probe tests
```

## License

Project code is MIT licensed; see `LICENSE`. Datasets and model weights remain under their own licenses and attribution requirements, described above and in the linked model cards.
