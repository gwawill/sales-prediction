from pathlib import Path

import pytest

from sales_poc.config import ConfigError, Settings, load_settings

REQUIRED = {
    "AWS_REGION": "us-east-1",
    "S3_BUCKET": "my-lake",
    "S3_VECTOR_BUCKET": "my-vectors",
    "KAFKA_BOOTSTRAP_SERVERS": "kafka-x.aivencloud.com:12345",
    "NEO4J_URI": "neo4j+s://abc.databases.neo4j.io",
    "NEO4J_PASSWORD": "neo-secret",
    "OPENROUTER_API_KEY": "or-secret",
    "LLM_MODEL": "some/model",
    "EMBEDDING_MODEL": "some/embedding",
    "EMBEDDING_DIM": "1024",
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Make sure real environment variables on this machine don't leak into the tests."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


def write_env(tmp_path: Path, values: dict[str, str]) -> Path:
    path = tmp_path / ".env"
    path.write_text("\n".join(f"{k}={v}" for k, v in values.items()), encoding="utf-8")
    return path


def test_loads_from_env_file_with_defaults(tmp_path):
    settings = load_settings(_env_file=write_env(tmp_path, REQUIRED))

    assert settings.aws_region == "us-east-1"
    assert settings.embedding_dim == 1024
    # Defaults are filled in.
    assert settings.kafka_topic == "orders"
    assert settings.agent_max_steps == 6
    assert settings.kafka_ssl_ca == Path("secrets/ca.pem")
    assert settings.aws_profile is None


def test_secrets_are_hidden_when_printed(tmp_path):
    settings = load_settings(_env_file=write_env(tmp_path, REQUIRED))

    assert "or-secret" not in repr(settings)
    assert "neo-secret" not in str(settings)
    assert settings.openrouter_api_key.get_secret_value() == "or-secret"


def test_missing_settings_are_all_named_in_one_error(tmp_path):
    values = {k: v for k, v in REQUIRED.items() if k not in ("S3_BUCKET", "NEO4J_PASSWORD")}

    with pytest.raises(ConfigError) as exc:
        load_settings(_env_file=write_env(tmp_path, values))

    message = str(exc.value)
    assert "S3_BUCKET" in message
    assert "NEO4J_PASSWORD" in message
    assert ".env.example" in message


def test_empty_value_counts_as_missing(tmp_path):
    values = {**REQUIRED, "AWS_REGION": ""}

    with pytest.raises(ConfigError, match="missing: AWS_REGION"):
        load_settings(_env_file=write_env(tmp_path, values))


def test_invalid_value_is_reported(tmp_path):
    values = {**REQUIRED, "EMBEDDING_DIM": "abc"}

    with pytest.raises(ConfigError, match="invalid: EMBEDDING_DIM"):
        load_settings(_env_file=write_env(tmp_path, values))


def test_environment_variable_overrides_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("KAFKA_TOPIC", "orders-test")

    settings = load_settings(_env_file=write_env(tmp_path, REQUIRED))

    assert settings.kafka_topic == "orders-test"
