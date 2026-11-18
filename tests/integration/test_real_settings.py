import pytest

from sales_poc.config import ALL_GROUPS, load_settings


@pytest.mark.parametrize("group", ALL_GROUPS, ids=lambda g: g.__name__)
def test_real_env_file_has_every_required_setting(group):
    """Checks the real .env, one group at a time, so each failure names one service."""
    load_settings(group)
