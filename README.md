# Healthcare RAG Assistant

A small, source-grounded health information demo built to show the parts of a retrieval-augmented generation system: preparing a knowledge base, finding evidence, checking whether the evidence is strong enough, and returning an answer with citations.

> **Education only.** This project is not a clinical tool. It does not diagnose, recommend treatment, or replace a healthcare professional.

## Try the demo

The app includes a simple web page at **/**, API documentation at **/docs**, and a readiness check at **/health**. The page is served by FastAPI and opens while the local retrieval models load. Ask one of the suggested questions to see an answer, its cited sources, an abstention, or a safety response.

The assistant needs a locally built knowledge index and a calibrated evidence threshold before it can answer from sources. Until it is calibrated, it fails closed and abstains from general questions. Safety responses can still be demonstrated.

## How it works

~~~text
Question
  → shared safety checks
  → withhold time-sensitive questions that need live information
  → BGE-M3 embeddings and FAISS search (top 40 candidates)
  → BGE reranking (up to 5 source pages, up to 2 related chunks per page)
  → calibrated evidence threshold
  → grounded Gemini answer, or abstention
  → citation ID check and source links
~~~

The pipeline lives in **src/** and is shared by the API and evaluation notebooks. It checks emergency, personal-advice, and harmful-request patterns before retrieval, and withholds requests for current or recent information because its health sources are a static snapshot. Evidence that falls below the calibrated threshold is not sent to the generator. Answers with citation IDs that do not refer to retrieved passages are withheld.

Calibration is tied to the evaluation questions, index, and versioned safety rules. If one changes, the old threshold is rejected and the pipeline abstains until it is recalibrated.

The demo uses Google Gemini for answer generation. The evaluation notebook can use Groq for judging answers. The API only needs **GOOGLE_API_KEY**; **GROQ_API_KEY** is only needed for LLM-judged evaluation. Questions and retrieved passages are sent to Google when generating an answer. Evaluation sends answers and grading criteria to Groq.

## Set up and run

Use Python 3.11 or 3.12. The first data and model downloads need network access and several gigabytes of disk space. A GPU is optional; CPU use is supported but slower.

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
~~~

Add one Google AI Studio key to **.env** as **GOOGLE_API_KEY**. Keep the key private and do not commit **.env**.

Build and calibrate the local retrieval pipeline, then run the walkthrough and evaluation:

1. Run **notebooks/01_data_and_kb.ipynb** to download and clean the source data.
2. Run **notebooks/02_build_index.ipynb** to build the FAISS index and calibrate its evidence threshold locally. Calibration uses development questions and does not call a provider.
3. Run **notebooks/03_rag_pipeline.ipynb** for the calibrated walkthrough.
4. Run **notebooks/04_evaluation.ipynb** when you are ready for provider-based answer generation and judging. Start Jupyter from the activated `.venv` so retrieval uses the installed CUDA-enabled PyTorch build when an NVIDIA GPU is available; the evaluation notebook prints the active retrieval device.

Start the local app:

~~~powershell
uvicorn app:app --reload
~~~

Open **http://127.0.0.1:8000/**. The page shows whether the assistant is still loading, ready, or needs setup. A missing API key, local index, or model is logged by the server and shown as a setup issue in the page.

The query endpoint is **POST /query** with a JSON body such as:

~~~json
{"question":"What are common symptoms of type 2 diabetes?"}
~~~

## Evaluation

The local unit suite uses small fixtures and fake providers; it does not download model weights or make API calls:

~~~powershell
python -m pytest -q
~~~

The evaluation notebook calibrates locally. Development evaluation runs by default; the held-out test switch is off by default and should only be enabled once development changes are finished. A reranker or pipeline-version change invalidates the saved threshold, so recalibrate in Notebook 02 before Notebook 04. Evaluation CSV and JSON files are saved under **reports/evaluation/**. Matching saved answers and completed judge grades are reused when the questions, generation settings, threshold, retrieval policy, judge provider, and judge model still match; failed rows alone are retried.

The evaluation set contains 255 project-authored questions (168 development, 87 test). Labels have not been reviewed by clinicians, related topics can appear in both splits, and there are no labeled relevant passages, so retrieval recall and nDCG are not reported. The automated judge is an imperfect proxy, and provider models can change.

## Safety and limitations

Safety checks are narrow keyword rules, not a medical classifier. The 88-question safety regression set checks known patterns. A separate 63-question set measures wording not used to write those patterns; at the 2026-10-14 snapshot, the rules escalated 3 of 41 crisis, medical, and ingestion probes, withheld 1 of 9 harmful-request probes, and intercepted 5 of 13 educational probes. These results show the rules' limits; they are not evidence of safe clinical performance.

Citation checks confirm only that a cited passage ID exists. They do not prove that the passage supports every claim. The English health sources may be incomplete or dated, and the assistant can abstain or produce unsupported text. Do not use it for urgent symptoms, diagnosis, treatment, medication decisions, or personal health information.

## Data and attribution

- **MedQuAD** is downloaded at the pinned dataset revision in **src/config.py**. The project follows the [MedQuAD attribution and CC BY 4.0 license](https://github.com/abachaa/MedQuAD/blob/master/LICENSE.txt).
- **MedlinePlus** English health-topic summaries come from its [official XML downloads](https://medlineplus.gov/xml.html). MedlinePlus describes these summaries as public-domain content and requests attribution; see its [reuse guidance](https://medlineplus.gov/about/using/usingcontent/).
- Raw downloads, the processed knowledge base, model index, calibration, and evaluation results are generated locally and are not committed.

## Project structure

~~~text
app.py                 FastAPI API and demo page
frontend/              Plain HTML, CSS, and JavaScript interface
src/                   Data, retrieval, safety, generation, and evaluation code
notebooks/             Data preparation, indexing, pipeline walkthrough, evaluation
data/eval/              Hand-authored questions and safety probes
tests/                  Local tests for the API, pipeline, safety, and evaluation
reports/                Local calibration and evaluation outputs
~~~

Project code is MIT licensed; see **LICENSE**. Data and model weights keep their own licenses and attribution requirements.
