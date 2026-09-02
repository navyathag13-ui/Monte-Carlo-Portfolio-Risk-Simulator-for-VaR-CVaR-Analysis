"""
tests/test_simulation.py
========================
Unit tests for the Monte Carlo simulation engine and stress-test framework.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest

from portfolio import build_portfolio
from simulation import run_monte_carlo, run_all_horizons
from stress_testing import run_all_stress_scenarios, worst_scenario


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_portfolio():
    """A fast 3-asset portfolio for simulation tests."""
    np.random.seed(0)
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame(
        {
            "X": 100 * np.cumprod(1 + np.random.normal(0.0005, 0.012, n)),
            "Y": 200 * np.cumprod(1 + np.random.normal(0.0003, 0.018, n)),
            "Z":  80 * np.cumprod(1 + np.random.normal(0.0002, 0.010, n)),
        },
        index=dates,
    )
    return build_portfolio(["X", "Y", "Z"], [0.4, 0.35, 0.25], prices)


# ---------------------------------------------------------------------------
# Single-period simulation
# ---------------------------------------------------------------------------

class TestRunMonteCarlo:
    N = 5_000

    def test_output_shape_1d(self, small_portfolio):
        result = run_monte_carlo(small_portfolio, horizon=1, n_simulations=self.N, random_seed=1)
        assert result.portfolio_returns.shape == (self.N,)

    def test_output_shape_multi(self, small_portfolio):
        result = run_monte_carlo(small_portfolio, horizon=10, n_simulations=self.N, random_seed=2)
        assert result.portfolio_returns.shape == (self.N,)

    def test_no_paths_by_default(self, small_portfolio):
        result = run_monte_carlo(small_portfolio, horizon=5, n_simulations=100,
                                 full_paths=False, random_seed=3)
        assert result.paths is None
        assert result.cumulative_paths is None

    def test_paths_stored_when_requested(self, small_portfolio):
        result = run_monte_carlo(small_portfolio, horizon=10, n_simulations=200,
                                 full_paths=True, random_seed=4)
        assert result.paths is not None
        assert result.cumulative_paths is not None
        assert result.cumulative_paths.shape == (11, 200)  # horizon+1 rows

    def test_cumulative_paths_start_at_one(self, small_portfolio):
        result = run_monte_carlo(small_portfolio, horizon=5, n_simulations=100,
                                 full_paths=True, random_seed=5)
        assert np.allclose(result.cumulative_paths[0, :], 1.0)

    def test_reproducibility(self, small_portfolio):
        r1 = run_monte_carlo(small_portfolio, horizon=1, n_simulations=1000, random_seed=99)
        r2 = run_monte_carlo(small_portfolio, horizon=1, n_simulations=1000, random_seed=99)
        assert np.array_equal(r1.portfolio_returns, r2.portfolio_returns)

    def test_different_seeds_differ(self, small_portfolio):
        r1 = run_monte_carlo(small_portfolio, horizon=1, n_simulations=1000, random_seed=1)
        r2 = run_monte_carlo(small_portfolio, horizon=1, n_simulations=1000, random_seed=2)
        assert not np.array_equal(r1.portfolio_returns, r2.portfolio_returns)

    def test_returns_finite(self, small_portfolio):
        result = run_monte_carlo(small_portfolio, horizon=1, n_simulations=self.N, random_seed=6)
        assert np.all(np.isfinite(result.portfolio_returns))

    def test_mean_return_reasonable(self, small_portfolio):
        """Mean simulated 1-day return should be near the historical mean."""
        result = run_monte_carlo(small_portfolio, horizon=1, n_simulations=self.N, random_seed=7)
        hist_mu = small_portfolio.daily_return
        sim_mu  = float(np.mean(result.portfolio_returns))
        assert abs(sim_mu - hist_mu) < 3 * small_portfolio.daily_volatility / np.sqrt(self.N), (
            f"Simulated mean {sim_mu:.6f} far from historical mean {hist_mu:.6f}"
        )

    def test_cov_override_changes_vol(self, small_portfolio):
        """Passing a higher-vol covariance override should widen the distribution."""
        base  = run_monte_carlo(small_portfolio, horizon=1, n_simulations=self.N, random_seed=8)
        cov_stressed = small_portfolio.cov_matrix.values * 4.0   # 2× volatility
        stressed = run_monte_carlo(small_portfolio, horizon=1, n_simulations=self.N,
                                   random_seed=8, cov_override=cov_stressed)
        assert np.std(stressed.portfolio_returns) > np.std(base.portfolio_returns)


# ---------------------------------------------------------------------------
# All-horizons wrapper
# ---------------------------------------------------------------------------

class TestRunAllHorizons:
    def test_returns_all_keys(self, small_portfolio):
        horizons = {"1d": 1, "5d": 5}
        results = run_all_horizons(small_portfolio, horizons, n_simulations=500, random_seed=10)
        assert set(results.keys()) == {"1d", "5d"}

    def test_larger_horizon_wider_std(self, small_portfolio):
        """Longer horizon should produce wider distribution (more uncertainty)."""
        horizons = {"1d": 1, "20d": 20}
        results = run_all_horizons(small_portfolio, horizons, n_simulations=2000, random_seed=11)
        std_1d  = np.std(results["1d"].portfolio_returns)
        std_20d = np.std(results["20d"].portfolio_returns)
        assert std_20d > std_1d * 1.5


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

SIMPLE_SCENARIOS = {
    "VolShock": {
        "description": "2x volatility",
        "vol_multiplier": 2.0,
        "corr_multiplier": 1.0,
        "price_shocks": {},
    },
    "PriceShock": {
        "description": "Equity down 10%",
        "vol_multiplier": 1.5,
        "corr_multiplier": 0.80,
        "price_shocks": {"X": -0.10},
    },
}


class TestStressTesting:
    def test_scenario_table_shape(self, small_portfolio):
        df = run_all_stress_scenarios(
            portfolio=small_portfolio,
            scenarios=SIMPLE_SCENARIOS,
            confidence_levels=[0.95, 0.99],
            horizon=1,
            n_simulations=1000,
            random_seed=20,
        )
        # (1 baseline + 2 scenarios) × 2 confidence levels = 6 rows
        assert len(df) == 6

    def test_stressed_var_exceeds_baseline(self, small_portfolio):
        """Stress scenarios should generally produce higher VaR than baseline."""
        df = run_all_stress_scenarios(
            portfolio=small_portfolio,
            scenarios=SIMPLE_SCENARIOS,
            confidence_levels=[0.95],
            horizon=1,
            n_simulations=3000,
            random_seed=21,
        )

        def _parse_pct(s):
            return float(str(s).replace("%", "").strip()) / 100

        base_var = _parse_pct(df[df["Scenario"] == "Baseline"]["VaR (%)"].iloc[0])
        for _, row in df[df["Scenario"] != "Baseline"].iterrows():
            stressed_var = _parse_pct(row["VaR (%)"])
            assert stressed_var >= base_var * 0.8, (
                f"Scenario '{row['Scenario']}' VaR ({stressed_var:.4f}) "
                f"is unexpectedly below baseline ({base_var:.4f})."
            )

    def test_worst_scenario_returns_string(self, small_portfolio):
        df = run_all_stress_scenarios(
            portfolio=small_portfolio,
            scenarios=SIMPLE_SCENARIOS,
            confidence_levels=[0.95],
            horizon=1,
            n_simulations=1000,
            random_seed=22,
        )
        name = worst_scenario(df, confidence="95%", metric="CVaR (%)")
        assert isinstance(name, str)
        assert name in SIMPLE_SCENARIOS
