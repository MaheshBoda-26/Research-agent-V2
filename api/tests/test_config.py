"""Tests for api/config.py — the plan §11.4 row for config:

malformed int/float raises ConfigError naming the variable;
ARXIV_DELAY_SECONDS < 3.0 is refused; blend weights must sum to 1.0;
describe() redacts every *key* field.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from config import ARXIV_MIN_DELAY_SECONDS, PROJECT_ROOT, ConfigError, Settings, load_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """Pin a known-baseline environment so .env on disk cannot leak into tests."""
    for name in list(os.environ):
        if name.startswith(("LLM_", "ARXIV_", "RETRIEVAL_", "RERANK_", "STAGE_CAP_")) or name in {
            "NVIDIA_API_KEY",
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "S2_API_KEY",
            "OPENALEX_MAILTO",
            "DATABASE_PATH",
            "ALLOWED_ORIGINS",
            "API_AUTH_TOKEN",
            "EMBED_MODEL",
            "EMBED_DEVICE",
            "CROSS_ENCODER_MODEL",
            "UMAP_RANDOM_STATE",
            "MAX_PAPERS_IN_PROMPT",
            "DEBUG",
        }:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ARXIV_OFFLINE", "1")
    return monkeypatch


def test_defaults_are_sane() -> None:
    settings = Settings.from_env()
    assert settings.llm_provider == "openai"
    assert settings.llm_model == ""  # never a guessed default
    assert settings.arxiv_delay_seconds == 3.0
    assert settings.db_path == (PROJECT_ROOT / "data" / "landscapes.db").resolve()
    assert settings.rerank_blend_ce + settings.rerank_blend_llm == pytest.approx(1.0)


def test_malformed_int_names_the_variable(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("RETRIEVAL_MAX_RESULTS", "two hundred")
    with pytest.raises(ConfigError, match="RETRIEVAL_MAX_RESULTS must be an integer"):
        Settings.from_env()


def test_malformed_float_names_the_variable(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("RERANK_BLEND_CE", "lots")
    with pytest.raises(ConfigError, match="RERANK_BLEND_CE must be a number"):
        Settings.from_env()


def test_arxiv_delay_below_tou_floor_is_refused_via_env(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("ARXIV_DELAY_SECONDS", "1")
    with pytest.raises(ConfigError, match="ARXIV_DELAY_SECONDS must be >= 3.0"):
        Settings.from_env()


def test_arxiv_delay_floor_is_unsettable_even_directly() -> None:
    """The guard lives in __post_init__: no construction path can breach the ToU."""
    with pytest.raises(ConfigError, match="ARXIV_DELAY_SECONDS"):
        Settings(arxiv_delay_seconds=ARXIV_MIN_DELAY_SECONDS - 0.5)


def test_blend_weights_must_sum_to_one() -> None:
    settings = replace(Settings(), rerank_blend_ce=0.9, rerank_blend_llm=0.9)
    with pytest.raises(ConfigError, match="must sum to 1.0"):
        settings.validate()


def test_validate_requires_llm_key_and_model_by_default() -> None:
    settings = replace(Settings(), llm_provider="nim", llm_api_key="", llm_model="")
    with pytest.raises(ConfigError) as excinfo:
        settings.validate(require_llm=True)
    message = str(excinfo.value)
    assert "NVIDIA_API_KEY" in message
    assert "LLM_MODEL" in message


def test_validate_ignores_llm_when_not_required() -> None:
    Settings().validate(require_llm=False)

def test_describe_redacts_every_key_and_token_field() -> None:
    settings = Settings(
        llm_api_key="nvapi-secret-value",
        s2_api_key="s2-secret-value",
        api_auth_token="token-secret-value",
    )
    described = settings.describe()
    assert described["llm_api_key"] == "set"
    assert described["s2_api_key"] == "set"
    assert described["api_auth_token"] == "set"
    leaked = [n for n, v in described.items() if isinstance(v, str) and "secret" in v]
    assert leaked == [], f"secret values leaked into describe(): {leaked}"
    # Non-secret fields stay visible — the point of describe() is a useful log line.
    assert described["llm_provider"] == "openai"
    assert described["arxiv_delay_seconds"] == 3.0


def test_llm_api_key_alias_resolution(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("LLM_PROVIDER", "nim")
    clean_env.setenv("NVIDIA_API_KEY", "nvapi-from-provider-var")
    assert Settings.from_env().llm_api_key == "nvapi-from-provider-var"
    clean_env.setenv("LLM_API_KEY", "generic-wins")
    assert Settings.from_env().llm_api_key == "generic-wins"


def test_llm_base_url_defaults_per_provider(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("LLM_PROVIDER", "nim")
    assert Settings.from_env().llm_base_url == "https://integrate.api.nvidia.com/v1"
    clean_env.setenv("LLM_BASE_URL", "http://localhost:11434/v1/")
    assert Settings.from_env().llm_base_url == "http://localhost:11434/v1"


def test_unknown_provider_is_a_typo_not_a_default(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("LLM_PROVIDER", "nmi")
    with pytest.raises(ConfigError, match="LLM_PROVIDER must be one of"):
        Settings.from_env()


def test_rerank_top_count_alias(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("RERANK_TOP_COUNT", "42")
    assert Settings.from_env().rerank_final_count == 42
    clean_env.setenv("RERANK_FINAL_COUNT", "55")
    assert Settings.from_env().rerank_final_count == 55


def test_cors_origins_split_and_strip() -> None:
    settings = Settings(allowed_origins="http://localhost:3000, https://example.com ,,")
    assert settings.cors_origins == ["http://localhost:3000", "https://example.com"]


def test_rerank_final_count_capped() -> None:
    settings = Settings(rerank_final_count=60, retrieval_max_results=50)
    assert settings.rerank_final_count_capped == 50


def test_hf_home_points_into_repo_at_import() -> None:
    hf_home = os.environ.get("HF_HOME", "")
    assert hf_home == str(PROJECT_ROOT / "data" / "models")
    st_home = os.environ.get("SENTENCE_TRANSFORMERS_HOME", "")
    assert st_home == str(PROJECT_ROOT / "data" / "models" / "sentence_transformers")


def test_stage_caps_match_the_plan_budgets() -> None:
    """§13.2 hard caps, encoded once, here."""
    settings = Settings()
    assert settings.stage_cap_retrieval_seconds == 30
    assert settings.stage_cap_enrichment_seconds == 60
    assert settings.stage_cap_rerank_seconds == 90
    assert settings.stage_cap_extraction_seconds == 300
    assert settings.stage_cap_layout_seconds == 60
    assert settings.stage_cap_synthesis_seconds == 120


def test_load_settings_makes_dirs(tmp_path, clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("DATABASE_PATH", str(tmp_path / "sub" / "dir" / "landscapes.db"))
    clean_env.setenv("RETRIEVAL_CACHE_DIR", str(tmp_path / "cache"))
    settings = load_settings(require_llm=False)
    assert settings.db_path.parent.is_dir()
    assert settings.retrieval_cache_dir.is_dir()
    assert settings.model_profile_dir.is_dir()

