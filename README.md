# Healthcare RAG Assistant

A source-grounded health information assistant that demonstrates practical AI engineering: data ingestion, vector retrieval, evidence-based abstention, citation validation, safety rules, and honest evaluation. This is an educational portfolio project, not a clinical system. It does not diagnose, prescribe, or replace professional care.

## Architecture

```text
question → safety rules → emergency help, personal-advice redirect, or harmful-request refusal
  → embed and search the cleaned MedQuAD + MedlinePlus corpus with FAISS (top 20)
  → BGE reranker (top 5) → calibrated evidence threshold
  → weak evidence: abstain | sufficient evidence: grounded LLM answer
  → validate citation IDs → answer with cited sources
```

The FastAPI service and notebooks use the same implementation in `src/`. Emergency, high-risk personal-advice, and explicit harmful-request checks run before retrieval. The pipeline abstains until a threshold has been calibrated for the current evaluation data, index, and safety rules. Generated answers must cite retrieved passages; structurally invalid citations cause an abstention.

- Exact FAISS inner-product search over normalized vectors keeps retrieval simple for this corpus.
- Retrieval uses [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) embeddings and [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) cross-encoder scores. Embedding, reranker, and MedQuAD snapshots are pinned to immutable revisions. The index manifest records model revisions, score-sensitive library versions, and file hashes; incompatible indexes and calibration files are rejected.
- **Generation and judging use separate providers.** Gemini generates answers; Groq's GPT-OSS 120B judges them. Quotas are project/organization scoped, so this separates the workloads but does not remove provider limits. Each API key variable accepts comma-separated keys: on HTTP 401, that key is skipped; on HTTP 429 or an explicit quota rejection, it is cooled down and the next key is tried once. Keys from the same project or organization can share quota, so rotation may not add capacity. Credentials are never written to logs.
- Citation checks verify that IDs refer to supplied passages. They do not prove semantic support for every claim.

## Design choices

I reviewed the [OpenAI Knowledge Retrieval starter](https://github.com/openai/openai-knowledge-retrieval), Haystack's [RAG evaluation example](https://github.com/deepset-ai/haystack/blob/main/e2e/pipelines/test_evaluation_pipeline.py), and the Sentence Transformers [retrieve-and-rerank example](https://github.com/huggingface/sentence-transformers/tree/main/examples/sentence_transformer/applications/retrieve_rerank). The useful choices here are inspectable pipeline stages, a two-stage retriever for a corpus too large for a cross-encoder scan, and explicit failure-case evaluation. This project keeps one shared Python pipeline and a small local evaluation harness instead of adding a framework, hosted vector store, or separate frontend. Retrieval recall is left unreported until relevant passages are labeled.

## Data

- **MedQuAD**: downloads the `lavita/MedQuAD` Hugging Face dataset at the revision in `src/config.py`. The original [MedQuAD project](https://github.com/abachaa/MedQuAD) describes the collection and publishes a [CC BY 4.0 license](https://github.com/abachaa/MedQuAD/blob/master/LICENSE.txt); rows without answer text are excluded and source publishers are retained.
- **MedlinePlus**: English health-topic summaries from the official [XML downloads](https://medlineplus.gov/xml.html). MedlinePlus identifies these summaries as public-domain content and requests source attribution; see its [reuse guidance](https://medlineplus.gov/about/using/usingcontent/). The download is a dated snapshot rather than an immutable source revision; the manifest records its URL and checksum. Other MedlinePlus material can have separate copyright terms and is not included in this project's stated data license.
- **Evaluation**: 255 hand-curated questions in `data/eval/questions.csv` (168 dev, 87 test). The project owner authored labels and key facts; clinicians have not reviewed them. Related topics can occur across splits, so the test set is only a small held-out check. There are no labeled relevant passages; retrieval recall@k and nDCG are not reported.

Downloaded sources, the processed knowledge base, index, calibration, and evaluation outputs are generated locally and not committed. See `data/eval/README.md` for label definitions and caveats.

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

   Run the setup cell and section 1, then stop if you only need calibration. This loads the retrieval models and saves the threshold without provider API keys or generation calls.

4. The optional pipeline walkthrough is `jupyter lab notebooks/03_rag_pipeline.ipynb`. Live generation is disabled by default; opt in for one answer-generation step when the evidence threshold allows it. Each request tries configured keys at most once when authentication or quota errors occur. Transient 429 responses can trigger up to five bounded retries and honor provider retry delays; evaluation stops when all configured keys hit a hard quota limit. Emergency, personal-advice, and explicit harmful-request examples do not call the model.
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

Run `jupyter lab notebooks/04_evaluation.ipynb`. Its first section calibrates on dev retrieval scores without API keys; to calibrate only, stop after section 1. Continuing initializes Gemini and Groq clients, runs the dev RAG evaluation, and requires their API keys. Gemini generates the answers, and Groq GPT-OSS 120B grades them with a separate judge prompt. `RUN_LLM_BASELINE` is off by default because it makes additional API calls. Set `RUN_TEST = True` only after code and the dev threshold are frozen; then run the notebook top to bottom once for the held-out split.

The notebook saves per-question answers, fact grades, behavior grades, judge coverage, citation-reference checks, and a summary under `reports/`. Answers are saved before LLM grading begins. One judge grading operation covers key facts and expected behaviour for each answer. Malformed JSON is marked as a grading error instead of triggering another judge generation; transient 429s still use bounded retries. If generation or judging stops on a provider limit, completed work is saved and remaining questions are marked skipped. Failed, skipped, and ungraded rows are reported separately and excluded from answer-quality denominators; check coverage before interpreting metrics. The LLM-only comparison is shown only when both systems complete generation and judging for the full same split.

Measures include key-fact coverage (LLM-judged), expected-behavior pass rates, abstention recall/precision, over-refusal, citation-ID validity, latency, and run coverage/failures. The threshold is selected on dev to meet a minimum abstention-recall target for insufficient-evidence questions with retrieval scores. This is a small, noisy calibration set, not a safety guarantee. The judge is an imperfect proxy, labels have not been clinician-reviewed, and safety rules were developed while inspecting this set. Gemini and Groq model IDs and behavior can change on the provider side; reports record IDs, but they are not immutable model snapshots. Do not infer clinical performance or causal improvement from these metrics. No metrics are prefilled here.

## Tests

The unit suite uses fake clients and small local data. It needs no model weights, GPU, generated index, or API credentials.

```powershell
python -m pip install -r requirements-test.txt
python -m pytest -q
```

GitHub Actions runs the same suite on pushes and pull requests.

## API and limitations

- `POST /query` accepts `{"question": "..."}` and returns an answer, emergency/redirect/refusal/abstention flags, sources actually cited, and a general-information disclaimer.
- `GET /health` reports readiness and whether calibration matches the current data, index, and models.

The synchronous API is intended for a local demo, not production traffic. Safety checks are narrow keyword rules and can miss rephrasings; the fixed crisis contacts are limited to selected regions plus an international directory. The English NIH consumer-health corpus may be incomplete or dated. The system can abstain when useful content is missing and can still produce unsupported claims despite structural citation checks. Do not submit personal or sensitive health information.

The crisis message currently lists Egypt's Ministry of Health line (**16328**), the US **988** Lifeline, the UK Samaritans line (**116 123**), and [Find A Helpline](https://findahelpline.com). Check the current information from the [Egyptian Ministry of Health](https://www.mohp.gov.eg/NewsDetails.aspx?subject_id=5901), [988 Lifeline](https://988lifeline.org/contact-us/), and [Samaritans](https://www.samaritans.org/how-we-can-help/contact-samaritan/) before relying on it. Generation sends the question and retrieved passages to Google Gemini; evaluation sends generated answers and grading criteria to Groq.

## Repository map

```text
app.py                         FastAPI endpoints
src/config.py                  paths, model snapshots, calibration validation
src/kb.py                      cleaning, chunking, embeddings, FAISS index + manifest
src/retriever.py               dense retrieval and cross-encoder reranking
src/safety.py                  emergency and personal-advice rules
src/generator.py               Gemini grounded prompt, generation, citation validation
src/provider_clients.py        Gemini generator and Groq judge clients
src/retry.py                   bounded provider rate-limit handling
src/pipeline.py                shared safety/retrieval/abstention/generation flow
src/calibration.py             dev-only threshold calibration
src/evaluation.py              batching, scoring, judge, and report helpers
notebooks/                     data, index, pipeline walkthrough, evaluation
data/eval/                     hand-authored labels and split definitions
tests/                         local unit and API behavior tests
```

## License

Project code is MIT licensed; see `LICENSE`. Datasets and model weights remain under their own licenses and attribution requirements, described above and in the linked model cards.
