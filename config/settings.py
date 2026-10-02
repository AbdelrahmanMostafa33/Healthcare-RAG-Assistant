from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    FAISS_INDEX_PATH: str = "data/embeddings/faiss_index/pubmedqa_index_flatip.faiss"
    CHUNKS_PKL_PATH: str = "data/embeddings/faiss_index/chunk_mapping.pkl"

    CLASSIFIER_PATH: str = "models/classifier/biobert_classifier"
    HF_CLASSIFIER_REPO: str = "AbdoMatrix/biobert-medical-classifier"

    EMBEDDING_MODEL: str = "pritamdeka/S-PubMedBert-MS-MARCO"

    LLM_MODEL: str = "openai/gpt-oss-120b"
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    JUDGE_MODEL: str = "llama-3.1-70b-versatile"

    USE_RERANKER: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    TOP_K: int = 30
    INJECT_K: int = 5
    MAX_CONTEXT_WORDS: int = 250
    MAX_TOKENS: int = 2048

    BM25_THRESHOLD: float = 12.0
    CATEGORY_EXPANSION: str = ""

    CORS_ORIGINS: list = ["*"]
    API_KEY: str = ""
    GROQ_API_KEY: str = ""
    HF_TOKEN: str = ""


    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()