from sales_poc.config import load_settings


def test_real_env_file_has_every_required_setting():
    """Checks the real .env in the project folder. Fails with the names of anything missing."""
    settings = load_settings()

    assert settings.s3_bucket
    assert settings.kafka_bootstrap_servers
