# Healthcare RAG Assistant

A Retrieval-Augmented Generation (RAG) system for answering medical questions using a biomedical knowledge base.

The system combines semantic and keyword retrieval with query classification and LLM generation to produce grounded medical answers.

## Architecture

```text
User Question
      |
      v
Query Classification
      |
      v
+-------------------+
| Hybrid Retrieval  |
|                   |
| FAISS + BM25      |
+---------+---------+
          |
          v
      Reranking
          |
          v
      LLM Generation
          |
          v
        Answer
```

## Main Components

* **Query Classification**: assigns the question to one of the medical categories.
* **FAISS**: semantic retrieval using biomedical embeddings.
* **BM25**: keyword-based retrieval for exact medical terms.
* **Reranking**: reorders retrieved candidates before passing them to the LLM.
* **LLM**: generates the final answer using the retrieved context.

## Dataset

The main knowledge base is built from the `pqa_artificial` subset of PubMedQA.

| Item        | Value                                                           |
| ----------- | --------------------------------------------------------------- |
| Dataset     | PubMedQA                                                        |
| Subset      | `pqa_artificial`                                                |
| Size        | ~211K records                                                   |
| Main fields | question, context, answer                                       |
| Categories  | Symptoms, Diagnosis, Treatment, Medication, Prevention, General |

The evaluation data is kept separately from the knowledge base to avoid using evaluation questions during retrieval.

## Models

### Query Classifier

* Base model: `dmis-lab/biobert-v1.1`
* Task: medical question classification
* Classes: 6 medical categories

### Embeddings

* Model: `pritamdeka/S-PubMedBert-MS-MARCO`
* Used to create embeddings for semantic retrieval
* Vectors are stored in FAISS

### Reranker

* Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
* Used to rerank retrieved candidates before generation

### Generator

* Model: `openai/gpt-oss-120b`
* Provider: Groq
* Interface: OpenAI-compatible API

The model configuration is kept in `config/settings.py` so it does not need to be repeated throughout the project.

## Project Structure

```text
Healthcare-RAG-Assistant/
│
├── api/
│   ├── main.py
│   ├── middleware/
│   ├── routes/
│   └── schemas/
│
├── config/
│   └── settings.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
│       └── faiss_index/
│
├── models/
│   └── classifier/
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_category_labelling.ipynb
│   ├── 05_embeddings_vectorstore.ipynb
│   ├── 06_rag_pipeline.ipynb
│   ├── 08_evaluation.ipynb
│   ├── 09_integrated_pipeline.ipynb
│   └── 10_end_to_end_test.ipynb
│
├── src/
│   ├── classification/
│   │   └── classifier.py
│   │
│   ├── data/
│   │   ├── hub.py
│   │   ├── labeller.py
│   │   ├── loader.py
│   │   └── preprocessor.py
│   │
│   ├── evaluation/
│   │   └── metrics.py
│   │
│   └── rag/
│       ├── bm25_retriever.py
│       ├── embeddings.py
│       └── vectorstore.py
│
├── tests/
│
├── docker/
│   └── Dockerfile
│
├── download.py
├── requirements.txt
├── setup.py
├── pytest.ini
├── .env.example
└── README.md
```

## Setup

Clone the repository:

```bash
git clone https://github.com/AbdelrahmanMostafa33/Healthcare-RAG-Assistant.git
cd Healthcare-RAG-Assistant
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Add the required environment variables to `.env`:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

Download the project data:

```bash
python download.py
```

## Running the API

Start the FastAPI application with:

```bash
uvicorn api.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

### Example Request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What are the symptoms of diabetes?\"}"
```

A typical response contains the generated answer, predicted category, and retrieved source information.

## Evaluation

Evaluation data is stored separately from the training and retrieval data.

```text
data/eval/questions.csv
```

The evaluation file contains separate `dev` and `test` splits.

* **dev**: used during development to compare configurations and improve the pipeline.
* **test**: kept untouched until the final evaluation.

The evaluation workflow is implemented in:

```text
notebooks/08_evaluation.ipynb
```

The current development process compares a plain LLM baseline against the RAG pipeline using the same evaluation questions.

## Notebooks

The notebooks cover the main development steps:

```text
01 -> Load the raw data

02 -> Clean and preprocess the data

03 -> Assign medical categories

05 -> Build embeddings and the FAISS index

06 -> Build and test the RAG pipeline

08 -> Run the evaluation experiments

09 -> Integrate the classifier and RAG pipeline

10 -> End-to-end verification
```

## Docker

The project also includes a Dockerfile for running the API in a container.

Build the image:

```bash
docker build -f docker/Dockerfile -t healthcare-rag .
```

Run it:

```bash
docker run -p 8000:8000 --env-file .env healthcare-rag
```

## Medical Disclaimer

This project is for educational and research purposes only.

It is not a substitute for professional medical advice, diagnosis, or treatment. Medical decisions should always be made with a qualified healthcare professional.

## License

MIT License
