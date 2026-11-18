"""Settings for the project, read from environment variables and the `.env` file.

Settings are split into groups, one per service. Each group is loaded only when the code that
needs it asks for it, so you can test S3 before you have filled in Kafka or Neo4j values.

    from sales_poc.config import get_aws_settings
    aws = get_aws_settings()
    aws.s3_bucket

Every setting is listed in `.env.example`. A setting with no default is required for its group.
If any are missing, the loader raises a `ConfigError` that names all of them at once.

Secret values use `SecretStr`, so they show as '**********' if printed or logged.
Call `.get_secret_value()` only at the point where the real value is needed.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Raised when settings are missing or invalid."""


class _GroupSettings(BaseSettings):
    """Shared behaviour for every settings group."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # "AWS_REGION=" with nothing after it counts as missing, not as an empty string.
        env_ignore_empty=True,
        # The .env file holds every group's values; each group ignores the others.
        extra="ignore",
    )


class AwsSettings(_GroupSettings):
    """S3 data lake. AWS keys are not read here: boto3 finds them itself
    (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, or a profile in ~/.aws)."""

    aws_region: str
    aws_profile: str | None = None
    s3_bucket: str


class VectorStoreSettings(_GroupSettings):
    """Amazon S3 Vectors (uses the AWS region and credentials from AwsSettings)."""

    s3_vector_bucket: str
    s3_vector_index: str = "sales-summaries"


class KafkaSettings(_GroupSettings):
    """Aiven for Apache Kafka.

    Two ways to log in:
      - username and password (SASL), used when KAFKA_USERNAME and KAFKA_PASSWORD are set;
      - client certificate files, used otherwise.
    Both need the CA certificate (KAFKA_SSL_CA) to check the server is really Aiven.
    """

    kafka_bootstrap_servers: str
    kafka_topic: str = "orders"
    kafka_consumer_group: str = "sales-poc-etl"
    kafka_ssl_ca: Path = Path("secrets/ca.pem")
    # SASL login
    kafka_username: str | None = None
    kafka_password: SecretStr | None = None
    kafka_sasl_mechanism: Literal["SCRAM-SHA-256", "SCRAM-SHA-512", "PLAIN"] = "SCRAM-SHA-256"
    # Client certificate login
    kafka_ssl_cert: Path = Path("secrets/service.cert")
    kafka_ssl_key: Path = Path("secrets/service.key")

    @property
    def uses_sasl(self) -> bool:
        return bool(self.kafka_username and self.kafka_password)


class Neo4jSettings(_GroupSettings):
    """Neo4j AuraDB."""

    neo4j_uri: str
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr


class LlmSettings(_GroupSettings):
    """OpenRouter language model and the AI agent's limits."""

    openrouter_api_key: SecretStr
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str
    agent_max_steps: int = Field(default=6, gt=0)


class EmbeddingSettings(_GroupSettings):
    """The model that turns text into embeddings (decided in task T1.5)."""

    embedding_provider: Literal["openrouter", "bedrock"] = "openrouter"
    embedding_model: str
    embedding_dim: int = Field(gt=0)


class AppSettings(_GroupSettings):
    """Local data and pipeline tuning. Everything here has a default."""

    data_dir: Path = Path("data")
    random_seed: int = 42
    stream_batch_size: int = Field(default=100, gt=0)
    stream_batch_seconds: float = Field(default=10.0, gt=0)


ALL_GROUPS: tuple[type[_GroupSettings], ...] = (
    AwsSettings,
    VectorStoreSettings,
    KafkaSettings,
    Neo4jSettings,
    LlmSettings,
    EmbeddingSettings,
    AppSettings,
)

S = TypeVar("S", bound=_GroupSettings)


def load_settings(group: type[S], **overrides) -> S:
    """Build one settings group, turning pydantic's error into one short, readable message."""
    try:
        return group(**overrides)
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
            f"{group.__name__} problem. "
            + "; ".join(parts)
            + ". See .env.example for every setting."
        ) from None


# Each getter loads its group once and then reuses it.


@lru_cache(maxsize=1)
def get_aws_settings() -> AwsSettings:
    return load_settings(AwsSettings)


@lru_cache(maxsize=1)
def get_vector_store_settings() -> VectorStoreSettings:
    return load_settings(VectorStoreSettings)


@lru_cache(maxsize=1)
def get_kafka_settings() -> KafkaSettings:
    return load_settings(KafkaSettings)


@lru_cache(maxsize=1)
def get_neo4j_settings() -> Neo4jSettings:
    return load_settings(Neo4jSettings)


@lru_cache(maxsize=1)
def get_llm_settings() -> LlmSettings:
    return load_settings(LlmSettings)


@lru_cache(maxsize=1)
def get_embedding_settings() -> EmbeddingSettings:
    return load_settings(EmbeddingSettings)


@lru_cache(maxsize=1)
def get_app_settings() -> AppSettings:
    return load_settings(AppSettings)
