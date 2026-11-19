import numpy as np
import pytest

from sales_poc.forecast.models import MODELS, SEASON_WEEKS, forecast

RNG = np.random.default_rng(0)


def steady(n=52, level=100.0, noise=5.0):
    return level + RNG.normal(0, noise, n)


def trending(n=52, start=100.0, step=3.0, noise=2.0):
    return start + step * np.arange(n) + RNG.normal(0, noise, n)


def seasonal(n=52, level=100.0, spike=80.0, noise=2.0):
    """A spike in the last 2 weeks of every 13-week quarter, like the quarter-end rush."""
    weeks = np.arange(n)
    return level + spike * (weeks % SEASON_WEEKS >= SEASON_WEEKS - 2) + RNG.normal(0, noise, n)


@pytest.mark.parametrize("model", MODELS)
def test_every_model_gives_four_non_negative_weeks(model):
    result = forecast(steady(), model, horizon=4)
    assert result.ok, result.note
    assert len(result.values) == 4
    assert (result.values >= 0).all()


@pytest.mark.parametrize("model", MODELS)
def test_a_steady_series_is_forecast_near_its_level(model):
    result = forecast(steady(noise=1.0), model)
    assert np.allclose(result.values, 100, atol=8)


def test_baseline_is_the_average_of_the_last_four_weeks():
    history = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=float)
    assert list(forecast(history, "baseline", horizon=3).values) == [65.0, 65.0, 65.0]


def test_holt_follows_a_trend_that_simple_smoothing_misses():
    history = trending()
    truth = 100 + 3 * np.arange(52, 56)

    holt_error = np.abs(forecast(history, "holt").values - truth).mean()
    simple_error = np.abs(forecast(history, "simple").values - truth).mean()

    assert holt_error < simple_error


def test_holt_winters_learns_the_quarter_end_spike():
    history = seasonal(n=50)  # weeks 50, 51 are the next spike (50 % 13 = 11, 51 % 13 = 12)
    hw = forecast(history, "holt_winters", horizon=4).values
    simple = forecast(history, "simple", horizon=4).values

    assert hw[0] > hw[2] + 40 and hw[1] > hw[2] + 40  # spike, then back to normal
    truth = np.array([180, 180, 100, 100])
    assert np.abs(hw - truth).mean() < np.abs(simple - truth).mean()


def test_holt_winters_needs_two_seasons_of_data():
    result = forecast(steady(n=2 * SEASON_WEEKS - 1), "holt_winters")
    assert not result.ok and "26 weeks" in result.note
    assert forecast(steady(n=2 * SEASON_WEEKS), "holt_winters").ok


def test_too_little_data_for_anything():
    for model in MODELS:
        assert not forecast(np.array([1.0, 2.0]), model).ok


def test_forecasts_never_go_below_zero():
    falling = np.maximum(200 - 10 * np.arange(30), 0).astype(float)
    for model in MODELS:
        assert (forecast(falling, model, horizon=8).values >= 0).all()


def test_an_all_zero_series_forecasts_zero():
    for model in MODELS:
        result = forecast(np.zeros(52), model)
        assert result.ok and np.allclose(result.values, 0, atol=1e-6)


def test_unknown_model_is_a_programming_error():
    with pytest.raises(ValueError, match="Unknown model"):
        forecast(steady(), "magic")
