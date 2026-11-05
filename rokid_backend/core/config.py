"""Application settings. Every value can be set through environment variables
prefixed with ROKID_ (e.g. ROKID_ADMIN_PASSWORD) or a .env file."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Domain hint placed inside the photo-transcription prompt. Override with ROKID_TRANSCRIBE_CONTEXT ("" disables it).
DEFAULT_TRANSCRIBE_CONTEXT = (
    "The questions come from the Romanian ANR exam for recreational boat operators (conducator de ambarcatiune de "
    "agrement, categories C and D): navigation rules (COLREG, R.N.D.), lights and sound signals, boat handling and "
    "construction, legislation, first aid. They are written in Romanian with diacritics (a, a, i, s, t with marks) and "
    "normally have exactly three options, A, B and C, which are often very short (for example 'da.', 'nu.', "
    "'nu este precizat.'). Copy numbers, units and abbreviations exactly (m, Nd, kW, CP, Hz, COLREG, R.N.D.)."
)


# Domain hint for answering unknown questions (documentation + model knowledge, web search). Override with ROKID_ANSWER_CONTEXT.
DEFAULT_ANSWER_CONTEXT = (
    "The questions come from the Romanian ANR exam for recreational boat operators (conducator de ambarcatiune de "
    "agrement, categories C and D): COLREG 1972, inland navigation rules (R.N.D.), Romanian legislation and sanctions, "
    "seamanship, boat construction and handling, lights and signals, buoyage, first aid, meteorology. The expected answer "
    "is the one the official Romanian regulations give (legislatie.just.ro, ANR materials), not a general opinion."
)


class Settings(BaseSettings):
    """Prefixed ROKID_* variables are canonical. A few conventional names are
    accepted as aliases so an existing .env keeps working: DATABASE_URL,
    OPENAI_API_KEY, OPENAI_MODEL, ANTHROPIC_API_KEY."""

    model_config = SettingsConfigDict(env_prefix="ROKID_", env_file=".env", extra="ignore", populate_by_name=True)

    app_name: str = "Rokid AI RAG Backend"
    version: str = "1.0.0"
    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"

    # Storage
    data_dir: Path = Path("rokid_data")
    database_url: str = Field(default="", validation_alias=AliasChoices("ROKID_DATABASE_URL", "DATABASE_URL"))  # default: sqlite in data_dir
    chroma_dir: Path | None = None  # default: data_dir / "chroma"

    # Authentication
    admin_username: str = "admin"
    admin_password: str = ""        # plain text or "pbkdf2_sha256$<iterations>$<salt>$<hash>"
    admin_api_token: str = ""       # optional static bearer token with admin scope
    device_tokens: str = ""         # comma separated initial device tokens (least privilege)
    session_secret: str = ""        # generated and persisted when empty
    session_hours: int = 12
    cookie_secure: bool = False     # set True behind HTTPS

    # Limits
    max_upload_mb: int = 50
    max_body_mb: int = 32          # any request body above this is refused with 413 (Rokid photo: up to ~4 MB in base64)
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 20
    idempotency_ttl_seconds: int = 600   # same request_id within this window returns the stored reply

    # Ingestion
    chunk_size: int = 900
    chunk_overlap: int = 150
    ocr_enabled: bool = True
    ocr_language: str = "eng+ron"

    # Embeddings
    embedding_provider: Literal["hash", "chroma_default", "openai"] = "hash"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"

    # Retrieval
    top_k_chunks: int = 6
    top_k_questions: int = 5
    semantic_question_threshold: float = 0.90   # cosine similarity to trust a verified record directly
    semantic_question_evidence_floor: float = 0.60  # below this verified records are not shown to the LLM
    min_chunk_similarity: float = 0.0
    option_match_threshold: float = 0.86
    option_match_margin: float = 0.08

    # LLM
    llm_provider: Literal["mock", "openai", "openai_compatible", "anthropic"] = "mock"
    llm_model: str = ""
    llm_api_key: str = ""
    openai_api_key: str = Field(default="", validation_alias=AliasChoices("ROKID_OPENAI_API_KEY", "OPENAI_API_KEY"))
    openai_model: str = Field(default="", validation_alias=AliasChoices("ROKID_OPENAI_MODEL", "OPENAI_MODEL"))
    anthropic_api_key: str = Field(default="", validation_alias=AliasChoices("ROKID_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"))
    llm_base_url: str = ""
    llm_timeout_seconds: float = 45.0
    llm_max_tokens: int = 700
    allow_general_knowledge: bool = False
    llm_reasoning_effort: str = "low"      # GPT-5/GPT-6/o-series: none|minimal|low|medium|high|xhigh; "" = not sent; auto-downgraded on HTTP 400
    llm_service_tier: str = ""             # OpenAI service_tier passthrough for lower latency (costs more); "" = not sent

    # Vision: photo -> question text, before the verified Q&A lookup
    vision_model: str = ""                 # model for photo transcription; default: the LLM model
    vision_detail: Literal["auto", "low", "high", "original"] = "high"
    vision_max_side: int = 1792            # photos whose longer side exceeds this (px) are downscaled before the call; 0 = never
    vision_reasoning_effort: str = "none"  # fastest setting on reasoning models; auto-downgraded when a model rejects it
    transcribe_context: str = DEFAULT_TRANSCRIBE_CONTEXT
    vision_crop: Literal["auto", "off"] = "auto"      # find the bright screen/page in the photo and send only it, at full resolution
    vision_crop_model_fallback: bool = True            # when the detector finds no screen, ask the vision model where it is (one extra small call)
    vision_min_sharpness: float = 25.0                 # Laplacian variance of the image sent (1200 px copy); below it the photo is refused as blurry; 0 = off

    # Unknown questions (not in the verified sets): documentation + model knowledge, and web search in parallel
    answer_grounding: Literal["strict", "open"] = "open"   # strict: only the uploaded evidence; open: evidence first, model knowledge allowed
    single_answer: bool = True             # exam questions have exactly one correct option; multi-option answers count as unsure
    answer_context: str = DEFAULT_ANSWER_CONTEXT
    docs_timeout_seconds: float = 15.0     # how long the pipeline waits for the documentation answer
    web_search_enabled: bool = True        # OpenAI Responses API web_search, run in parallel with the documentation answer (openai provider only)
    web_search_model: str = "gpt-6.1-terra"
    web_search_reasoning_effort: str = "low"
    web_search_context_size: Literal["low", "medium", "high"] = "medium"
    web_search_timeout_seconds: float = 8.0
    web_search_country: str = "RO"
    web_search_allowed_domains: str = ""   # comma separated (max 100); empty = the whole web
    web_search_blocked_domains: str = ""
    web_search_prefer: bool = True         # when documentation and web disagree, show the web answer (marked "Verifica")
    candidates_enabled: bool = True        # keep non-verified answers for review in admin (confirm -> verified set)

    cors_origins: list[str] = Field(default_factory=list)

    # Antrenor CAA (study web app served by the same server)
    caa_enabled: bool = True
    caa_data_dir: Path = PROJECT_ROOT / "date"
    caa_auto_download: bool = False   # fetch the ANR PDFs at startup when a category is missing

    @property
    def llm_key(self) -> str:
        if self.llm_api_key:
            return self.llm_api_key
        return self.anthropic_api_key if self.llm_provider == "anthropic" else self.openai_api_key

    @property
    def llm_model_name(self) -> str:
        if self.llm_model:
            return self.llm_model
        if self.llm_provider in ("openai", "openai_compatible") and self.openai_model:
            return self.openai_model
        return {"openai": "gpt-4o-mini", "anthropic": "claude-sonnet-5-5"}.get(self.llm_provider, "")

    @property
    def vision_model_name(self) -> str:
        return self.vision_model or self.llm_model_name

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "rokid.sqlite3"

    @property
    def effective_database_url(self) -> str:
        return self.database_url or f"sqlite:///{self.sqlite_path}"

    @property
    def effective_chroma_dir(self) -> Path:
        return self.chroma_dir or (self.data_dir / "chroma")

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def load_settings(**overrides) -> Settings:
    return Settings(**overrides)
