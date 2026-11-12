"""Application configuration."""

import os
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(BASE_DIR, ".env"))

INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
os.makedirs(INSTANCE_DIR, exist_ok=True)


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
    INSTANCE_DIR = INSTANCE_DIR
    DATABASE = os.path.join(INSTANCE_DIR, "documents.db")
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(INSTANCE_DIR, 'documents.db')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(INSTANCE_DIR, "uploads")
    MODEL_CACHE_DIR = os.environ.get("EMBEDDING_CACHE_DIR", os.path.join(INSTANCE_DIR, "model_cache"))
    EMBEDDING_OFFLINE = os.environ.get("EMBEDDING_OFFLINE", "0") == "1"

    # Classroom limits
    MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MiB
    MAX_PDF_PAGES = 50
    MAX_WORKSHEETS = 50
    MAX_WORKBOOK_CELLS = 50000
    MAX_INDEXED_PASSAGES = 500

    ALLOWED_EXTENSIONS = {"pdf", "docx", "html", "htm", "txt", "md", "markdown", "xlsx", "xls", "pptx", "ppt"}

    # Embedding model (FastEmbed)
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIM = 384
    # Model has 128 token max. 2 special tokens (<s> and </s>) are added,
    # leaving 126 content tokens. We target ~100 tokens per passage with
    # ~20-token overlap so we never risk silent truncation.
    PASSAGE_MAX_TOKENS = 100
    PASSAGE_OVERLAP_TOKENS = 20

    # Local Ollama configuration
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "")
    OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "60"))
