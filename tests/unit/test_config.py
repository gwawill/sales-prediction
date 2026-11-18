from pathlib import Path

import pytest

from sales_poc.config import (
    ALL_GROUPS,
    AppSettings,
    AwsSettings,
    ConfigError,
    EmbeddingSettings,
    KafkaSettings,
    LlmSettings,
    Neo4jSettings,
    VectorStoreSettings,
    load_settings,
)

# The required values for each group, nothing more.
REQUIRED = {
    AwsSettings: {"AWS_REGION": "us-east-1", "S3_BUCKET": "my-lake"},
    VectorStoreSettings: {"S3_VECTOR_BUCKET": "my-vectors"},
    KafkaSettings: {"KAFKA_BOOTSTRAP_SERVERS": "kafka-x.aivencloud.com:12345"},
    Neo4jSettings: {"NEO4J_URI": "neo4j+s://abc.databases.neo4j.io", "NEO4J_PASSWORD": "neo-secret"},
    LlmSettings: {"OPENROUTER_API_KEY": "or-secret", "LLM_MODEL": "some/model"},
    EmbeddingSettings: {"EMBEDDING_MODEL": "some/embedding", "EMBEDDING_DIM": "1024"},
    AppSettings: {},
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Make sure real environment variables on this machine don't leak into the tests."""
    for group in ALL_GROUPS:
        for name in group.model_fields:
            monkeypatch.delenv(name.upper(), raising=False)


def write_env(tmp_path: Path, values: dict[str, str]) -> Path:
    path = tmp_path / ".env"
    path.write_text("\n".join(f"{k}={v}" for k, v in values.items()), encoding="utf-8")
    return path


def test_every_group_is_covered_by_these_tests():
    assert set(REQUIRED) == set(ALL_GROUPS)


@pytest.mark.parametrize("group", ALL_GROUPS, ids=lambda g: g.__name__)
def test_each_group_loads_with_only_its_own_values(tmp_path, group):
    # The point of the split: e.g. AWS loads even when Kafka, Neo4j and LLM values are absent.
    load_settings(group, _env_file=write_env(tmp_path, REQUIRED[group]))


@pytest.mark.parametrize(
    "group", [g for g in ALL_GROUPS if REQUIRED[g]], ids=lambda g: g.__name__
)
def test_each_group_names_all_its_missing_values(tmp_path, group):
    with pytest.raises(ConfigError) as exc:
        load_settings(group, _env_file=write_env(tmp_path, {}))

    message = str(exc.value)
    assert group.__name__ in message
    for name in REQUIRED[group]:
        assert name in message
    assert ".env.example" in message


def test_defaults_are_filled_in(tmp_path):
    env = write_env(tmp_path, {**REQUIRED[KafkaSettings], **REQUIRED[LlmSettings]})

    kafka = load_settings(KafkaSettings, _env_file=env)
    llm = load_settings(LlmSettings, _env_file=env)
    app = load_settings(AppSettings, _env_file=env)
    aws = load_settings(AwsSettings, _env_file=write_env(tmp_path, REQUIRED[AwsSettings]))

    assert kafka.kafka_topic == "orders"
    assert kafka.kafka_ssl_ca == Path("secrets/ca.pem")
    assert llm.agent_max_steps == 6
    assert app.random_seed == 42
    assert aws.aws_profile is None


def test_a_group_ignores_other_groups_values(tmp_path):
    everything = {k: v for values in REQUIRED.values() for k, v in values.items()}

    aws = load_settings(AwsSettings, _env_file=write_env(tmp_path, everything))

    assert not hasattr(aws, "kafka_bootstrap_servers")


def test_secrets_are_hidden_when_printed(tmp_path):
    env = write_env(tmp_path, {**REQUIRED[Neo4jSettings], **REQUIRED[LlmSettings]})

    neo4j = load_settings(Neo4jSettings, _env_file=env)
    llm = load_settings(LlmSettings, _env_file=env)

    assert "neo-secret" not in str(neo4j)
    assert "or-secret" not in repr(llm)
    assert llm.openrouter_api_key.get_secret_value() == "or-secret"


def test_empty_value_counts_as_missing(tmp_path):
    values = {**REQUIRED[AwsSettings], "AWS_REGION": ""}

    with pytest.raises(ConfigError, match="missing: AWS_REGION"):
        load_settings(AwsSettings, _env_file=write_env(tmp_path, values))


def test_invalid_value_is_reported(tmp_path):
    values = {**REQUIRED[EmbeddingSettings], "EMBEDDING_DIM": "abc"}

    with pytest.raises(ConfigError, match="invalid: EMBEDDING_DIM"):
        load_settings(EmbeddingSettings, _env_file=write_env(tmp_path, values))


def test_environment_variable_overrides_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("KAFKA_TOPIC", "orders-test")

    kafka = load_settings(KafkaSettings, _env_file=write_env(tmp_path, REQUIRED[KafkaSettings]))

    assert kafka.kafka_topic == "orders-test"
