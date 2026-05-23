from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    GEMINI_API_KEY: str = ""
    LLM_PROVIDER: str = "groq"            # "groq" | "gemini" | "ollama" | "openai"
    OLLAMA_MODEL: str = "mistral"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama3-8b-8192"   # free, fast, no rate-limit issues

    # Embedding
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector DB
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "rag_docs"

    # Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # Retrieval
    TOP_K: int = 5
    RERANK_ENABLED: bool = True

    # PDF ingestion
    PDF_DIR: str = "./data/pdfs"
    OCR_ENABLED: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
