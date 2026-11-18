"""Settings for the whole project, read from environment variables and the `.env` file.

Every setting is listed in `.env.example`. Settings with no default are required: if any are
missing, `get_settings()` raises a `ConfigError` that names all of them at once.

Secret values use `SecretStr`, so they show as '**********' if printed or logged.
Call `.get_secret_value()` only at the point where the real value is needed.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Raised when settings are missing or invalid."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # "AWS_REGION=" with nothing after it counts as missing, not as an empty string.
        env_ignore_empty=True,
        extra="ignore",
    )

    # --- AWS: S3 data lake and S3 Vectors ---
    # AWS credentials themselves are NOT read here. boto3 finds them on its own
    # (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, or a profile in ~/.aws).
    aws_region: str
    aws_profile: str | None = None
    s3_bucket: str
    s3_vector_bucket: str
    s3_vector_index: str = "sales-summaries"

    # --- Aiven Kafka ---
    kafka_bootstrap_servers: str
    kafka_topic: str = "orders"
    kafka_consumer_group: str = "sales-poc-etl"
    kafka_ssl_ca: Path = Path("secrets/ca.pem")
    kafka_ssl_cert: Path = Path("secrets/service.cert")
    kafka_ssl_key: Path = Path("secrets/service.key")

    # --- Neo4j AuraDB ---
    neo4j_uri: str
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr

    # --- OpenRouter (LLM) and embeddings ---
    openrouter_api_key: SecretStr
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str
    embedding_provider: Literal["openrouter", "bedrock"] = "openrouter"
    embedding_model: str
    embedding_dim: int = Field(gt=0)

    # --- AI agent ---
    agent_max_steps: int = Field(default=6, gt=0)

    # --- Local data and pipelines ---
    data_dir: Path = Path("data")
    random_seed: int = 42
    stream_batch_size: int = Field(default=100, gt=0)
    stream_batch_seconds: float = Field(default=10.0, gt=0)


def load_settings(**overrides) -> Settings:
    """Build settings, turning pydantic's error into one short, readable message."""
    try:
        return Settings(**overrides)
    except ValidationError as exc:
        missing = [
            str(err["loc"][0]).upper() for err in exc.errors() if err["type"] == "missing"
        ]
        invalid = [
            f"{str(err['loc'][0]).upper()} ({err['msg']})"
            for err in exc.errors()
            if err["type"] != "missing"
        ]
        parts = []
        if missing:
            parts.append("missing: " + ", ".join(missing))
        if invalid:
            parts.append("invalid: " + ", ".join(invalid))
        raise ConfigError(
            "Settings problem. " + "; ".join(parts) + ". See .env.example for every setting."
        ) from None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The settings for this process, loaded once and then reused."""
    return load_settings()
