"""Runtime configuration, read from environment variables and an optional `.env` file."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings in one place. Every LLM parameter can be overridden from `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM endpoint: any OpenAI-compatible API (NVIDIA build, vLLM, Ollama, ...).
    llm_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_api_key: SecretStr = SecretStr("")
    # Deliberately empty: the model slug must be chosen with `pi models`, never guessed.
    llm_model: str = ""
    llm_temperature: float = 0.2
    llm_max_concurrency: int = 2
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 5
    llm_max_tokens: int = 2048
    # Price in USD per million tokens, for cost estimates in evaluation reports. The NVIDIA
    # API catalog is free for prototyping, hence 0 by default; set them for paid endpoints.
    llm_price_input_per_mtok: float = 0.0
    llm_price_output_per_mtok: float = 0.0

    # Local storage.
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")
    cache_dir: Path = Path(".cache")

    # Last date of the generated history. Fixed so that the database is reproducible.
    data_end_date: date = date(2026, 10, 30)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "portfolio.duckdb"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
