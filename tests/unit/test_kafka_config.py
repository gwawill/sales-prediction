import pytest

from sales_poc.config import KafkaSettings
from sales_poc.kafka import KafkaSetupError, connection_config, login_method, missing_ssl_files


def settings(tmp_path, create=("ca.pem", "service.cert", "service.key"), **extra):
    for name in create:
        (tmp_path / name).write_text("x", encoding="utf-8")
    return KafkaSettings(
        _env_file=None,
        kafka_bootstrap_servers="kafka-x.aivencloud.com:12345",
        kafka_ssl_ca=tmp_path / "ca.pem",
        kafka_ssl_cert=tmp_path / "service.cert",
        kafka_ssl_key=tmp_path / "service.key",
        **extra,
    )


def test_certificate_login_uses_ssl_and_the_three_files(tmp_path):
    config = connection_config(settings(tmp_path))

    assert config["bootstrap.servers"] == "kafka-x.aivencloud.com:12345"
    assert config["security.protocol"] == "SSL"
    assert config["ssl.ca.location"].endswith("ca.pem")
    assert config["ssl.certificate.location"].endswith("service.cert")
    assert config["ssl.key.location"].endswith("service.key")
    assert "sasl.username" not in config


def test_missing_certificate_files_are_named(tmp_path):
    s = settings(tmp_path, create=("ca.pem",))

    assert [m.split("=")[0] for m in missing_ssl_files(s)] == ["KAFKA_SSL_CERT", "KAFKA_SSL_KEY"]
    with pytest.raises(KafkaSetupError, match="KAFKA_SSL_CERT.*KAFKA_SSL_KEY.*secrets/"):
        connection_config(s)


def test_username_and_password_switch_to_sasl(tmp_path):
    s = settings(tmp_path, create=("ca.pem",), kafka_username="avnadmin", kafka_password="pw")

    config = connection_config(s)

    assert config["security.protocol"] == "SASL_SSL"
    assert config["sasl.mechanism"] == "SCRAM-SHA-256"
    assert config["sasl.username"] == "avnadmin"
    assert config["sasl.password"] == "pw"
    assert config["ssl.ca.location"].endswith("ca.pem")
    # Certificate files are not needed, so their absence is not an error.
    assert "ssl.certificate.location" not in config
    assert login_method(s) == "SASL SCRAM-SHA-256"


def test_sasl_still_needs_the_ca_file(tmp_path):
    s = settings(tmp_path, create=(), kafka_username="avnadmin", kafka_password="pw")

    with pytest.raises(KafkaSetupError, match="KAFKA_SSL_CA"):
        connection_config(s)


def test_username_without_password_falls_back_to_certificate(tmp_path):
    s = settings(tmp_path, kafka_username="avnadmin")

    assert connection_config(s)["security.protocol"] == "SSL"
    assert login_method(s) == "client certificate"


def test_password_is_hidden_when_settings_are_printed(tmp_path):
    s = settings(tmp_path, kafka_username="avnadmin", kafka_password="super-secret")

    assert "super-secret" not in repr(s)
