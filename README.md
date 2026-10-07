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
| **In scope** | What a condition is, its symptoms, causes and risk factors, prevention, tests and diagnosis, and how it is generally treated |
| **Partly in scope** | Medicines, but only as far as the sources describe them (for example the kinds of medicine used for a condition). There is no drug-specific source, so questions like "what are the side effects of metformin?" will often get an abstention. |
| **Out of scope** | Diagnosing the user, personal advice, doses, whether to start, stop or combine medicines, interpreting someone's test results, live or current information |

## How it works

```
User question
  -> embed the question                                    (BAAI/bge-m3)
  -> FAISS: the 20 closest chunks                          (exact inner-product search)
  -> cross-encoder rerank, keep the best 5                 (ms-marco-MiniLM-L-12-v2)
  -> check evidence: best rerank score >= threshold ?
        no  -> abstain (no LLM call)
        yes -> grounded prompt with numbered passages -> one LLM call   (gpt-oss-120b on Groq)
  -> answer with [1], [2] citations + the sources used + disclaimer
```

| Step | Why it is there |
|---|---|
| Dense retrieval | Finds chunks about the same thing as the question even when the wording differs. Fast. |
| Rerank | The cross-encoder reads the question and a chunk together, so it judges relevance better than comparing two separate embeddings. Too slow for the whole index, so it only sees the 20 candidates. |
| Evidence check | Retrieval always returns something. Without a check, the LLM would be asked to answer from irrelevant text. |
| Grounded prompt | Numbered passages and strict rules (use only these, cite them, say what is missing, no personal advice) make answers checkable. |
| Sources | Only the passages that were in the prompt are returned, with source, title and URL. |

There is no query classification, no second retrieval pass, no retry with a looser prompt and no fallback to the model's own knowledge.

## Knowledge base

| Source | Content |
|---|---|
| [MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD) | Question/answer pairs written by NIH websites (NIDDK, CDC, NHLBI, ...). Kept with each document's URL and publisher so answers can be cited. |
| [MedlinePlus](https://medlineplus.gov/xml.html) | One plain-language summary per health topic (NLM). |

Documents are cleaned (HTML and NIH site text removed, duplicates dropped) and split into chunks of at most 1000 characters made of whole sentences, with at most one sentence (200 characters) repeated between chunks. The document's question is placed on top of every chunk so it keeps its topic. Chunk size and overlap are reasonable defaults, not tuned values.

**Why a small knowledge base.** The sources were chosen to match the questions, not to be large. PubMedQA (research abstracts with generated questions) and MedRedQA (informal Reddit replies) were left out because they fit consumer questions poorly. This is a judgement about fit; their effect on retrieval was not measured.

**The medication gap.** `lavita/MedQuAD` ships without the answers of its drug, A.D.A.M., herbal and GARD subsets (31,034 rows, removed for copyright), and MedlinePlus drug pages are AHFS content copyrighted by ASHP, so they are not used. The open alternative is FDA labels (DailyMed / openFDA), which are written for clinicians and need their own ingestion step that was not built. So medication is limited to what the health-topic sources say, and the assistant abstains otherwise. Notebook 01 shows the gap in the data; notebook 04 reports medication questions separately.

The index is not stored in the repository. Notebooks 01 and 02 build it.

## Tech stack

Python, pandas, BeautifulSoup, sentence-transformers (embeddings and reranker), FAISS, the Groq API through the OpenAI SDK, FastAPI with Pydantic, pytest, Jupyter.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # with a GPU, install the matching torch build first
cp .env.example .env                 # then put your GROQ_API_KEY in .env
```

1. Build the knowledge base and index by running `notebooks/01_data_and_kb.ipynb`, then `notebooks/02_build_index.ipynb`. A GPU is recommended for the embedding step.
2. Start the API and ask a question:

```bash
uvicorn app:app
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the symptoms of type 2 diabetes?"}'
```

Interactive docs are at `http://localhost:8000/docs`. Startup loads the models, so it takes a little while.

3. Run the tests. They use small fakes, so no models, GPU or API key are needed (run on Python 3.12):

```bash
pytest
```

## API

`POST /query` with `{"question": "..."}` (3 to 1000 characters) returns:

| Field | Meaning |
|---|---|
| `answer` | The answer with `[n]` citations, or the abstention message |
| `abstained` | `true` when the sources did not contain enough relevant evidence |
| `sources` | The passages used: `n` (matches the citation), `source`, `title`, `url`. Empty when abstaining. |
| `disclaimer` | The medical disclaimer |

`GET /health` returns the status, the abstention threshold in use, and whether that threshold has been calibrated.

## Notebooks

Read them in order. Each states the question it answers, and they are shipped without outputs.

| Notebook | What it shows |
|---|---|
| `01_data_and_kb` | Scope, sources and why, cleaning, the final knowledge base, and a check of how well it covers the scope |
| `02_build_index` | Chunking, embeddings, the FAISS index, and checks that the index is correct |
| `03_rag_pipeline` | One question walked through every step by hand: embed, FAISS, rerank, evidence check, prompt, LLM call. Then the same thing through `RAGPipeline` |
| `04_evaluation` | Retrieval, groundedness and safety behaviour, RAG vs an LLM-only baseline, and how the abstention threshold is chosen |

## Evaluation

**Questions.** `data/eval/questions.csv` has 150 questions, split into dev (100) and test (50). Consumer, research, medication and ambiguous questions have key facts; unanswerable, emergency, refuse and safety-redirect questions have an expected behaviour.

**Metrics** (all computed by notebook 04, graded by a judge model that is not the generator):

| Layer | Metrics |
|---|---|
| Retrieval | Key-fact Recall@5 and MRR: FAISS only vs FAISS + reranker, and chunks with vs without the question line |
| Generation | Key-fact coverage; share of answer claims supported by the passages the model was given |
| Behaviour | Pass rate for answer, emergency, refuse, safety-redirect and insufficient-evidence questions; abstention recall and precision; over-refusal; latency |

**Abstention threshold.** Chosen on the dev split as the value with the best balanced accuracy between answering answerable questions and abstaining on unanswerable ones. Medication questions are left out of that choice because the knowledge base cannot support them. The test split is run once, with the threshold frozen. Until notebook 04 has been run, `src/config.py` uses a placeholder of 0.0, and `/health` reports `threshold_calibrated: false`.

### Results

**Only the LLM-only baseline has been measured so far.** The RAG pipeline has not been evaluated: the repository ships without the index, the models and an API key, so there are no RAG numbers here and none have been written in. Running notebook 04 produces them and overwrites `reports/results.md`.

| Metric (dev, 100 questions) | LLM-only baseline | RAG |
|---|---|---|
| Key-fact coverage | 0.982 | not run yet |
| Behaviour pass: answer | 68/68 | not run yet |
| Behaviour pass: emergency | 7/7 | not run yet |
| Behaviour pass: insufficient evidence | 12/13 | not run yet |
| Behaviour pass: refuse | 3/3 | not run yet |
| Behaviour pass: safety redirect | 8/9 | not run yet |

The baseline is already near the ceiling because most questions concern common health topics an LLM knows well, so RAG is **not** expected to beat it on coverage. What it should add is answers that stay inside the sources, visible citations, and an explicit "not enough information" instead of a confident guess.

## Examples

**Abstention.** The response is deterministic; no LLM call is made:

```json
{
  "answer": "I couldn't find enough relevant information in my current sources to answer this reliably. Try rephrasing the question, or ask a doctor or pharmacist. If you think this could be an emergency, call your local emergency number or go to the nearest emergency department now.",
  "abstained": true,
  "sources": [],
  "disclaimer": "This is general health information for education only. It is not medical advice, diagnosis or treatment. Talk to a doctor or pharmacist about your own situation, and call your local emergency number if you think it is an emergency."
}
```

**Grounded answer: the response shape.** The text and sources depend on the knowledge base and the model. Notebook 03 prints real outputs.

```json
{
  "answer": "<answer using only the passages, with [1] [2] citations>",
  "abstained": false,
  "sources": [
    {"n": 1, "source": "MedlinePlus", "title": "<topic>", "url": "https://medlineplus.gov/..."},
    {"n": 2, "source": "MedQuAD (<publisher>)", "title": "<topic>", "url": "https://..."}
  ],
  "disclaimer": "..."
}
```

**Safety behaviour** is built in at two points. The system prompt tells the model to start with "contact emergency services" for possible emergency symptoms, and never to give a diagnosis, a dose, or advice to start, stop or change a medicine; it should point to a doctor or pharmacist instead. When the pipeline abstains, the fixed message also tells the user to call their local emergency number if they think it is an emergency. How well this works is measured by the emergency and safety-redirect checks in notebook 04, which have not been run for the RAG pipeline yet.

## Limitations

- **Educational use only.** No diagnosis, no personal advice, no doses. It knows nothing about the user.
- **Knowledge base.** NIH consumer-health content in English. It is a static snapshot: MedQuAD is 2019 data, so some content is dated, and there is no live information. There is no drug source, so medication answers are limited to what the health-topic sources say.
- **Retrieval limits the answer.** If the right passage is not retrieved the answer is incomplete, and if the threshold is too strict, answerable questions are refused.
- **Evaluation.** Key facts were written by the project owner and have not been reviewed by a clinician. Answers are graded by an LLM (a preview model); every verdict is saved. The dev split has only 13 unanswerable questions, so a threshold tuned on it is a noisy estimate. Most questions are answerable from common knowledge. The 20 medication questions expect answers the knowledge base mostly cannot give; they are reported separately and excluded from the threshold choice.
- **Not reproduced end to end here.** The index is generated, not shipped. The encoder, the chunk size and the question line on chunks were not compared against alternatives. Notebook 04 contains the question-line comparison, and it has not been run.

## What I tried and removed

An earlier version was reviewed and simplified. The evidence behind each decision differs, so it is stated:

| Removed | Reason | Evidence |
|---|---|---|
| Query classifier and category routing | Trained on labels from a keyword rule that agreed with hand labels on only about 55% of the 150 evaluation questions. In retrieval its categories were compared with chunk categories using a different vocabulary, so it could not change which chunks came back. It also added a model download and a hard dependency. | Label agreement measured; routing mismatch found by code review |
| BM25 / hybrid retrieval | Switched off by default while the documentation described hybrid search. Nothing in the repository showed a gain. It should come back only if it beats dense + rerank in notebook 04. | No measurement either way |
| Fallbacks that answer without evidence (the model's own knowledge, a looser retry prompt, returning a raw chunk) | They contradict the goal of answering only from sources. Replaced by abstention. | Design decision |
| PubMedQA and MedRedQA in the knowledge base | Poor fit for consumer questions, and they dominated the index by size. | Judgement, not measured |
| Chunks made of the question and nothing else | The earlier chunking produced 219,458 of them (23.2% of all chunks). Now the text is split first and the question added afterwards. | Measured |
| Key rotation, API auth, response cache, extra middleware, a static dashboard | Not useful for this project, and the dashboard showed metrics that no longer described the system. | Review |
| BLEU / ROUGE / BERTScore | Not meaningful for checking facts and groundedness. | Judgement |

## Project structure

```
app.py                  FastAPI service (POST /query, GET /health)
src/
  config.py             settings: paths, models, chunk size, k, threshold
  kb.py                 clean the sources, chunk, embed, build and load the FAISS index
  retriever.py          embed -> FAISS top 20 -> rerank -> top 5
  generator.py          grounded prompt and the single LLM call
  pipeline.py           RAGPipeline.answer(): retrieve -> evidence check -> generate -> sources
  evaluation.py         judge prompts, scoring, threshold choice, abstention metrics
notebooks/              01 data, 02 index, 03 pipeline, 04 evaluation
data/eval/questions.csv the 150 evaluation questions
reports/                results.md and the saved LLM-only baseline verdicts
tests/                  behaviour tests (no models or API key needed)
```

## License

MIT, see `LICENSE`.
