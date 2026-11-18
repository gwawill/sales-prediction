"""One place to build Kafka connection settings for Aiven, so the check, the producer and the
consumer all connect the same way.

Every connection is encrypted with SSL and checks the server against the CA certificate
(KAFKA_SSL_CA), which proves the server is really Aiven. Then we log in one of two ways:

  - Username and password (SASL), when KAFKA_USERNAME and KAFKA_PASSWORD are set.
    This is what the Aiven free plan accepts.
  - Client certificate, otherwise: KAFKA_SSL_CERT (who we are) and KAFKA_SSL_KEY (its
    private key).
"""

from __future__ import annotations

from sales_poc.config import KafkaSettings, get_kafka_settings


class KafkaSetupError(RuntimeError):
    """The Kafka settings or certificate files are not ready."""


def missing_ssl_files(settings: KafkaSettings) -> list[str]:
    """Names of the needed SSL files that do not exist on disk."""
    files = {"KAFKA_SSL_CA": settings.kafka_ssl_ca}
    if not settings.uses_sasl:
        files["KAFKA_SSL_CERT"] = settings.kafka_ssl_cert
        files["KAFKA_SSL_KEY"] = settings.kafka_ssl_key
    return [f"{name}={path}" for name, path in files.items() if not path.is_file()]


def connection_config(settings: KafkaSettings | None = None) -> dict[str, str]:
    """librdkafka settings shared by producers, consumers and admin clients."""
    s = settings or get_kafka_settings()
    missing = missing_ssl_files(s)
    if missing:
        raise KafkaSetupError(
            "Kafka SSL file(s) not found: "
            + ", ".join(missing)
            + ". Download them from the Aiven console into the secrets/ folder."
        )

    config = {
        "bootstrap.servers": s.kafka_bootstrap_servers,
        "ssl.ca.location": str(s.kafka_ssl_ca),
    }
    if s.uses_sasl:
        assert s.kafka_password is not None  # guaranteed by uses_sasl
        config |= {
            "security.protocol": "SASL_SSL",
            "sasl.mechanism": s.kafka_sasl_mechanism,
            "sasl.username": s.kafka_username,
            "sasl.password": s.kafka_password.get_secret_value(),
        }
    else:
        config |= {
            "security.protocol": "SSL",
            "ssl.certificate.location": str(s.kafka_ssl_cert),
            "ssl.key.location": str(s.kafka_ssl_key),
        }
    return config


def login_method(settings: KafkaSettings | None = None) -> str:
    """A short, safe description of how we log in (for messages and logs)."""
    s = settings or get_kafka_settings()
    return f"SASL {s.kafka_sasl_mechanism}" if s.uses_sasl else "client certificate"
