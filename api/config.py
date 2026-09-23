"""Central configuration for the research landscape agent (V2).

All tunables live here so pipeline behaviour can be adjusted without touching
pipeline logic. Ported from V1's ``api/config.py`` and extended with the
enrichment, graph, full-text and stage-budget knobs the V2 plan references
(Appendix A.1 is the contract this module implements).

Paths in the environment are resolved relative to the project root (the parent
of this file's directory) so that the CLI, the API server, and the tests all
agree on where ``data/`` lives regardless of the working directory they are
launched from.

``HF_HOME`` and ``SENTENCE_TRANSFORMERS_HOME`` are pointed at
``data/models/`` **at import time**, before anything in the stack can import
torch/transformers — otherwise model weights leak into the user's global
``~/.cache/huggingface`` and the project's footprint becomes invisible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

#: Where HuggingFace-weight caches live, anchored inside the repo.
DEFAULT_MODELS_DIR = PROJECT_ROOT / "data" / "models"

# Import-time guard: these must be in place before any torch/transformers
# import anywhere in the process. ``setdefault`` so an explicit operator
# setting still wins.
os.environ.setdefault("HF_HOME", str(DEFAULT_MODELS_DIR))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(DEFAULT_MODELS_DIR / "sentence_transformers"))


class ConfigError(RuntimeError):
    """Raised when the environment is missing something required to run."""


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _str(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _path(name: str, default: str) -> Path:
    """Resolve a path setting, relative entries anchored at the project root."""
    raw = _str(name, default)
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


VALID_PROVIDERS = ("openai", "openrouter", "nim", "local")

#: Default base URL per provider (LLM_BASE_URL overrides).
_PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "nim": "https://integrate.api.nvidia.com/v1",
    "local": "http://localhost:11434/v1",
}

#: Provider-specific key variable consulted when LLM_API_KEY is empty.
_PROVIDER_KEY_VARS = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "nim": "NVIDIA_API_KEY",
    "local": "",  # local servers (e.g. Ollama's shim) need no key
}

#: arXiv's Terms of Use floor: no more than one request every three seconds.
ARXIV_MIN_DELAY_SECONDS = 3.0


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the environment.

    Frozen so tests can derive isolated variants with ``dataclasses.replace``
    and so a running pipeline's configuration cannot drift mid-run.
    """

    # --- LLM ---
    llm_provider: str = "openai"  # openai | openrouter | nim | local
    llm_api_key: str = ""  # resolved per provider, never logged
    llm_base_url: str = ""
    llm_model: str = ""  # never a guessed default; verify-llm proves it
    llm_concurrency: int = 4  # openai | openrouter | nim | local
    llm_max_repairs: int = 2  # max JSON repair attempts per call
    llm_timeout_seconds: int = 120  # per-request HTTP timeout on the LLM

    # --- Retrieval (arXiv) ---
    arxiv_delay_seconds: float = 3.0  # __post_init__ refuses anything below 3.0
    arxiv_num_retries: int = 5
    arxiv_page_size: int = 100
    retrieval_max_results: int = 200
    retrieval_cache_ttl_hours: int = 24
    retrieval_cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "cache" / "arxiv")
    arxiv_offline: bool = False
    arxiv_force_429: bool = False

    # --- Enrichment ---
    s2_api_key: str = ""
    openalex_mailto: str = ""  # polite pool
    enrich_concurrency: int = 4
    enrich_ttl_days: int = 7
    citation_blend_weight: float = 0.15

    # --- Rerank ---
    rerank_seed_count: int = 40
    rerank_final_count: int = 60
    rerank_blend_ce: float = 0.6
    rerank_blend_llm: float = 0.4
    rerank_judge_batch_size: int = 10
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_device: str = "cpu"
    cross_encoder_max_length: int = 512
    cross_encoder_batch_size: int = 16

    # --- Extraction / layout ---
    prompt_version: str = "extract_v2"
    extract_timeout_seconds: int = 60
    max_papers_in_prompt: int = 30
    embed_model: str = "BAAI/bge-base-en-v1.5"
    embed_device: str = "cpu"
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.0
    umap_random_state: int = 42
    cluster_min_size_ratio: float = 0.06
    # HDBSCAN defaults min_samples to min_cluster_size — far too conservative
    # for a reading map (V1 measured 40% noise at the default). 3 is the
    # lowest value that kept well-separated structure intact in V1's tests.
    hdbscan_min_samples: int = 3

    # --- Graph ---
    graph_knn_k: int = 3
    graph_cosine_floor: float = 0.60
    graph_cosine_start: float = 0.72
    graph_edges_per_node: int = 6
    graph_min_density: float = 2.5

    # --- Full text (Phase 11) ---
    fulltext_enabled: bool = False
    fulltext_top_n: int = 15

    # --- Stage hard caps in seconds (plan §13.2) ---
    stage_cap_retrieval_seconds: int = 30
    stage_cap_enrichment_seconds: int = 60
    stage_cap_rerank_seconds: int = 90
    stage_cap_extraction_seconds: int = 300
    stage_cap_layout_seconds: int = 60
    stage_cap_synthesis_seconds: int = 120

    # --- API / storage / security ---
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "landscapes.db")
    api_port: int = 8000
    allowed_origins: str = "http://localhost:3000"
    api_auth_token: str = ""  # empty = open (localhost only)
    debug: bool = False
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        # The ToU floor is not a preference: make it impossible to construct a
        # Settings that breaches it, however the value arrived (env, .env, or a
        # direct constructor call in a test).
        if self.arxiv_delay_seconds < ARXIV_MIN_DELAY_SECONDS:
            raise ConfigError(
                f"ARXIV_DELAY_SECONDS must be >= {ARXIV_MIN_DELAY_SECONDS} per the arXiv Terms of Use "
                f"(no more than one request every three seconds); got {self.arxiv_delay_seconds}."
            )

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the process environment (and ``.env``)."""
        provider = _str("LLM_PROVIDER", "openai").lower()
        if provider not in VALID_PROVIDERS:
            # An unknown provider is a typo, not a request for the default.
            raise ConfigError(f"LLM_PROVIDER must be one of {list(VALID_PROVIDERS)}, got {provider!r}")
        key_var = _PROVIDER_KEY_VARS[provider]
        return cls(
            llm_provider=provider,
            llm_api_key=_str("LLM_API_KEY", "") or (_str(key_var, "") if key_var else ""),
            llm_base_url=_str("LLM_BASE_URL", "").rstrip("/") or _PROVIDER_BASE_URLS[provider],
            llm_model=_str("LLM_MODEL", ""),
            llm_concurrency=_int("LLM_CONCURRENCY", 4),
            llm_max_repairs=_int("LLM_MAX_REPAIRS", 2),
            llm_timeout_seconds=_int("LLM_TIMEOUT_SECONDS", 120),
            arxiv_delay_seconds=_float("ARXIV_DELAY_SECONDS", 3.0),
            arxiv_num_retries=_int("ARXIV_NUM_RETRIES", 5),
            arxiv_page_size=_int("ARXIV_PAGE_SIZE", 100),
            retrieval_max_results=_int("RETRIEVAL_MAX_RESULTS", 200),
            retrieval_cache_ttl_hours=_int("RETRIEVAL_CACHE_TTL_HOURS", 24),
            retrieval_cache_dir=_path("RETRIEVAL_CACHE_DIR", "./data/cache/arxiv"),
            arxiv_offline=_bool("ARXIV_OFFLINE", False),
            arxiv_force_429=_bool("ARXIV_FORCE_429", False),
            s2_api_key=_str("S2_API_KEY", ""),
            openalex_mailto=_str("OPENALEX_MAILTO", ""),
            enrich_concurrency=_int("ENRICH_CONCURRENCY", 4),
            enrich_ttl_days=_int("ENRICH_TTL_DAYS", 7),
            citation_blend_weight=_float("CITATION_BLEND_WEIGHT", 0.15),
            rerank_seed_count=_int("RERANK_SEED_COUNT", 40),
            # RERANK_TOP_COUNT is accepted as an alias (V1's .env used it).
            rerank_final_count=_int("RERANK_FINAL_COUNT", _int("RERANK_TOP_COUNT", 60)),
            rerank_blend_ce=_float("RERANK_BLEND_CE", 0.6),
            rerank_blend_llm=_float("RERANK_BLEND_LLM", 0.4),
            rerank_judge_batch_size=_int("RERANK_JUDGE_BATCH_SIZE", 10),
            cross_encoder_model=_str("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            cross_encoder_device=_str("CROSS_ENCODER_DEVICE", "cpu"),
            cross_encoder_max_length=_int("CROSS_ENCODER_MAX_LENGTH", 512),
            cross_encoder_batch_size=_int("CROSS_ENCODER_BATCH_SIZE", 16),
            prompt_version=_str("PROMPT_VERSION", "extract_v2"),
            extract_timeout_seconds=_int("EXTRACT_TIMEOUT_SECONDS", 60),
            max_papers_in_prompt=_int("MAX_PAPERS_IN_PROMPT", 30),
            embed_model=_str("EMBED_MODEL", "BAAI/bge-base-en-v1.5"),
            embed_device=_str("EMBED_DEVICE", "cpu"),
            umap_n_neighbors=_int("UMAP_N_NEIGHBORS", 30),
            umap_min_dist=_float("UMAP_MIN_DIST", 0.0),
            umap_random_state=_int("UMAP_RANDOM_STATE", 42),
            cluster_min_size_ratio=_float("CLUSTER_MIN_SIZE_RATIO", 0.06),
            hdbscan_min_samples=max(1, _int("HDBSCAN_MIN_SAMPLES", 3)),
            graph_knn_k=_int("GRAPH_KNN_K", 3),
            graph_cosine_floor=_float("GRAPH_COSINE_FLOOR", 0.60),
            graph_cosine_start=_float("GRAPH_COSINE_START", 0.72),
            graph_edges_per_node=_int("GRAPH_EDGES_PER_NODE", 6),
            graph_min_density=_float("GRAPH_MIN_DENSITY", 2.5),
            fulltext_enabled=_bool("FULLTEXT_ENABLED", False),
            fulltext_top_n=_int("FULLTEXT_TOP_N", 15),
            stage_cap_retrieval_seconds=_int("STAGE_CAP_RETRIEVAL_SECONDS", 30),
            stage_cap_enrichment_seconds=_int("STAGE_CAP_ENRICHMENT_SECONDS", 60),
            stage_cap_rerank_seconds=_int("STAGE_CAP_RERANK_SECONDS", 90),
            stage_cap_extraction_seconds=_int("STAGE_CAP_EXTRACTION_SECONDS", 300),
            stage_cap_layout_seconds=_int("STAGE_CAP_LAYOUT_SECONDS", 60),
            stage_cap_synthesis_seconds=_int("STAGE_CAP_SYNTHESIS_SECONDS", 120),
            db_path=_path("DATABASE_PATH", "./data/landscapes.db"),
            api_port=_int("API_PORT", 8000),
            allowed_origins=_str("ALLOWED_ORIGINS", "http://localhost:3000"),
            api_auth_token=_str("API_AUTH_TOKEN", ""),
            debug=_bool("DEBUG", False),
            log_level=_str("LOG_LEVEL", "INFO"),
        )



    # --- Derived ---

    @property
    def cors_origins(self) -> list[str]:
        """Explicit origin list; never ``*`` (invalid with credentials)."""
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @property
    def model_profile_dir(self) -> Path:
        """Local HuggingFace cache root, so model downloads land inside data/."""
        return self.db_path.parent / "models"

    @property
    def rerank_final_count_capped(self) -> int:
        """Never ask for more final papers than we retrieved."""
        return min(self.rerank_final_count, self.retrieval_max_results)

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.retrieval_cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_profile_dir.mkdir(parents=True, exist_ok=True)
        self.apply_model_cache_env()

    def apply_model_cache_env(self) -> None:
        """Point HuggingFace at ``data/models`` before any model is imported.

        Import-time ``setdefault`` in this module already covers the common
        case; this method re-asserts the anchored location for settings whose
        ``db_path`` was overridden (tests, ``--db`` flags). ``setdefault`` so
        an explicit environment setting still wins.
        """
        os.environ.setdefault("HF_HOME", str(self.model_profile_dir))
        os.environ.setdefault(
            "SENTENCE_TRANSFORMERS_HOME", str(self.model_profile_dir / "sentence_transformers")
        )

    def validate(self, *, require_llm: bool = True) -> None:
        """Fail early and name the missing variable.

        Tests run without an LLM key, so ``require_llm=False`` is supported.
        """
        problems: list[str] = []
        if require_llm and not self.llm_api_key and self.llm_provider != "local":
            var = _PROVIDER_KEY_VARS.get(self.llm_provider) or "LLM_API_KEY"
            problems.append(
                f"{var} is not set (LLM_PROVIDER={self.llm_provider}). "
                "Copy .env.example to .env and fill it in."
            )
        if require_llm and not self.llm_model:
            problems.append(
                "LLM_MODEL is empty. Run `make verify-llm` to pick a slug from the "
                "provider's live catalogue — never guess one."
            )
        if self.arxiv_delay_seconds < ARXIV_MIN_DELAY_SECONDS:
            problems.append(
                "ARXIV_DELAY_SECONDS must be >= 3.0 per the arXiv Terms of Use "
                "(no more than one request every three seconds)."
            )
        if self.rerank_final_count < 1:
            problems.append("RERANK_FINAL_COUNT must be at least 1.")
        blend = self.rerank_blend_ce + self.rerank_blend_llm
        if abs(blend - 1.0) > 1e-6:
            problems.append(f"RERANK_BLEND_CE + RERANK_BLEND_LLM must sum to 1.0, got {blend}.")
        if not 0.0 < self.rerank_blend_ce < 1.0:
            problems.append("RERANK_BLEND_CE must be strictly between 0 and 1.")
        if not 0.0 <= self.citation_blend_weight <= 1.0:
            problems.append("CITATION_BLEND_WEIGHT must be between 0 and 1.")
        if problems:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(problems))

    def describe(self) -> dict[str, object]:
        """Redacted view of the configuration — the only config object that may be logged."""
        out: dict[str, object] = {}
        for f in fields(self):
            value: object = getattr(self, f.name)
            if "key" in f.name or "token" in f.name:
                value = "set" if value else "unset"
            elif isinstance(value, Path):
                value = str(value)
            out[f.name] = value
        return out


def load_settings(*, require_llm: bool = False) -> Settings:
    """Load, validate, and materialize directories in one call."""
    settings = Settings.from_env()
    settings.validate(require_llm=require_llm)
    settings.ensure_dirs()
    return settings

