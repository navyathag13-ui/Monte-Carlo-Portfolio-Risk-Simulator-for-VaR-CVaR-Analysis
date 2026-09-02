"""
tests/test_risk_metrics.py
==========================
Unit tests for VaR and CVaR computations in risk_metrics.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from risk_metrics import compute_var, compute_cvar, distribution_stats, build_risk_table


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_returns():
    """10,000 standard-normal samples — known analytical properties."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(10_000)


@pytest.fixture
def skewed_returns():
    """10,000 samples with a heavy left tail (negative skew)."""
    rng = np.random.default_rng(7)
    base = rng.standard_normal(10_000) * 0.01
    # Add some large negative values
    tail = -rng.exponential(0.05, 200)
    all_rets = np.concatenate([base, tail])
    return all_rets


# ---------------------------------------------------------------------------
# VaR tests
# ---------------------------------------------------------------------------

class TestVaR:
    def test_var_positive(self, normal_returns):
        """VaR must be a positive number (represents a loss magnitude)."""
        var = compute_var(normal_returns, 0.95)
        assert var > 0, "VaR should be positive."

    def test_var_99_greater_than_95(self, normal_returns):
        """99% VaR must be larger than 95% VaR."""
        var_95 = compute_var(normal_returns, 0.95)
        var_99 = compute_var(normal_returns, 0.99)
        assert var_99 > var_95, "99% VaR should exceed 95% VaR."

    def test_var_mc_vs_parametric_close(self, normal_returns):
        """Monte Carlo and parametric VaR should be within 20% for normal data."""
        var_mc   = compute_var(normal_returns, 0.95, method="monte_carlo")
        var_para = compute_var(normal_returns, 0.95, method="parametric")
        relative_diff = abs(var_mc - var_para) / var_para
        assert relative_diff < 0.20, (
            f"MC VaR ({var_mc:.4f}) and parametric VaR ({var_para:.4f}) "
            f"diverge by {relative_diff:.1%}."
        )

    def test_var_95_approx_magnitude_standard_normal(self, normal_returns):
        """
        For unit-normal returns the 5th percentile is ~-1.645,
        so 95% VaR should be ~1.645 * daily_vol.
        Here daily_vol ≈ 1 so VaR_95 should be near 1.6 (with finite-sample noise).
        """
        var_95 = compute_var(normal_returns, 0.95)
        assert 1.4 < var_95 < 1.9, f"Unexpected VaR_95 for unit-normal: {var_95:.3f}"

    def test_var_constant_returns_zero(self):
        """Portfolio with no variance should have zero VaR."""
        flat = np.zeros(1000)
        var = compute_var(flat, 0.95)
        assert var == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# CVaR tests
# ---------------------------------------------------------------------------

class TestCVaR:
    def test_cvar_exceeds_var(self, normal_returns):
        """CVaR (expected shortfall) must always exceed VaR at same confidence."""
        for cl in [0.90, 0.95, 0.99]:
            var  = compute_var(normal_returns,  cl)
            cvar = compute_cvar(normal_returns, cl)
            assert cvar >= var, f"CVaR ({cvar:.4f}) < VaR ({var:.4f}) at {cl:.0%}."

    def test_cvar_99_greater_than_95(self, normal_returns):
        """99% CVaR must exceed 95% CVaR."""
        cvar_95 = compute_cvar(normal_returns, 0.95)
        cvar_99 = compute_cvar(normal_returns, 0.99)
        assert cvar_99 > cvar_95

    def test_cvar_positive(self, normal_returns):
        """CVaR must be positive."""
        cvar = compute_cvar(normal_returns, 0.95)
        assert cvar > 0

    def test_cvar_parametric(self, normal_returns):
        """Parametric CVaR should be consistent with MC CVaR for normal data."""
        cvar_mc   = compute_cvar(normal_returns, 0.95, method="monte_carlo")
        cvar_para = compute_cvar(normal_returns, 0.95, method="parametric")
        rel_diff = abs(cvar_mc - cvar_para) / cvar_para
        assert rel_diff < 0.25, (
            f"MC CVaR ({cvar_mc:.4f}) vs parametric ({cvar_para:.4f}) "
            f"differ by {rel_diff:.1%}."
        )

    def test_cvar_left_tail_skew(self, skewed_returns):
        """Heavy left-tail returns should show CVaR >> VaR."""
        var  = compute_var(skewed_returns,  0.95)
        cvar = compute_cvar(skewed_returns, 0.95)
        assert cvar > var * 1.05, "Expected CVaR to materially exceed VaR for skewed data."


# ---------------------------------------------------------------------------
# Distribution statistics tests
# ---------------------------------------------------------------------------

class TestDistributionStats:
    def test_keys_present(self, normal_returns):
        stats = distribution_stats(normal_returns)
        required = {"mean", "std", "skewness", "kurtosis", "min", "max",
                    "p1", "p5", "p25", "median", "p75", "p95", "p99"}
        assert required.issubset(stats.keys())

    def test_normal_skew_near_zero(self, normal_returns):
        stats = distribution_stats(normal_returns)
        assert abs(stats["skewness"]) < 0.15, "Skewness of normal should be ~0."

    def test_normal_kurt_near_zero(self, normal_returns):
        """Excess kurtosis of standard normal should be ~0."""
        stats = distribution_stats(normal_returns)
        assert abs(stats["kurtosis"]) < 0.30

    def test_percentile_ordering(self, normal_returns):
        stats = distribution_stats(normal_returns)
        assert stats["p1"] < stats["p5"] < stats["p25"] < stats["median"]
        assert stats["median"] < stats["p75"] < stats["p95"] < stats["p99"]


# ---------------------------------------------------------------------------
# Risk table tests
# ---------------------------------------------------------------------------

class TestBuildRiskTable:
    def test_shape(self, normal_returns):
        df = build_risk_table(
            mc_returns=normal_returns,
            hist_returns=normal_returns[:500],
            confidence_levels=[0.95, 0.99],
            portfolio_value=1_000_000,
        )
        # 3 methods × 2 confidence levels = 6 rows
        assert len(df) == 6
        assert "VaR (%)" in df.columns
        assert "CVaR (%)" in df.columns

    def test_no_hist_returns(self, normal_returns):
        df = build_risk_table(
            mc_returns=normal_returns,
            hist_returns=None,
            confidence_levels=[0.95],
            portfolio_value=1_000_000,
        )
        # 2 methods (MC + Parametric) × 1 confidence level = 2 rows
        assert len(df) == 2
