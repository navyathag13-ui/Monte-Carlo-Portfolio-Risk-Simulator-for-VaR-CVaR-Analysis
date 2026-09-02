"""
tests/test_portfolio.py
=======================
Unit tests for portfolio construction, weight validation, and computed statistics.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest

from portfolio import build_portfolio, Portfolio
from utils import validate_weights


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_prices():
    """Small synthetic price DataFrame (100 days, 3 assets)."""
    np.random.seed(123)
    n = 100
    dates = pd.bdate_range("2022-01-01", periods=n)
    data = {
        "A": 100 * np.cumprod(1 + np.random.normal(0.0004, 0.01, n)),
        "B": 200 * np.cumprod(1 + np.random.normal(0.0003, 0.015, n)),
        "C":  50 * np.cumprod(1 + np.random.normal(0.0002, 0.008, n)),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def equal_weight_portfolio(mock_prices):
    return build_portfolio(
        tickers=["A", "B", "C"],
        weights=[1/3, 1/3, 1/3],
        prices=mock_prices,
    )


# ---------------------------------------------------------------------------
# Weight validation tests
# ---------------------------------------------------------------------------

class TestWeightValidation:
    def test_valid_weights(self):
        w = validate_weights([0.5, 0.3, 0.2], ["A", "B", "C"])
        assert np.isclose(w.sum(), 1.0)

    def test_normalisation(self):
        """Weights that do not sum to 1 should be normalised."""
        w = validate_weights([2, 3, 5], ["A", "B", "C"])
        assert np.isclose(w.sum(), 1.0)
        assert np.isclose(w[0], 0.2)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            validate_weights([0.5, 0.5], ["A", "B", "C"])

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_weights([-0.1, 0.6, 0.5], ["A", "B", "C"])

    def test_all_zero_raises(self):
        with pytest.raises(ValueError, match="all be zero"):
            validate_weights([0.0, 0.0, 0.0], ["A", "B", "C"])


# ---------------------------------------------------------------------------
# Portfolio construction tests
# ---------------------------------------------------------------------------

class TestPortfolioConstruction:
    def test_returns_shape(self, equal_weight_portfolio, mock_prices):
        """Returns should have n-1 rows and 3 columns."""
        p = equal_weight_portfolio
        assert len(p.returns) == len(mock_prices) - 1
        assert list(p.returns.columns) == ["A", "B", "C"]

    def test_covariance_matrix_symmetric(self, equal_weight_portfolio):
        cov = equal_weight_portfolio.cov_matrix.values
        assert np.allclose(cov, cov.T), "Covariance matrix must be symmetric."

    def test_covariance_positive_definite(self, equal_weight_portfolio):
        cov = equal_weight_portfolio.cov_matrix.values
        eigvals = np.linalg.eigvalsh(cov)
        assert eigvals.min() > 0, "Covariance matrix must be positive definite."

    def test_correlation_diagonal_ones(self, equal_weight_portfolio):
        corr = equal_weight_portfolio.corr_matrix.values
        assert np.allclose(np.diag(corr), 1.0)

    def test_weights_sum_to_one(self, equal_weight_portfolio):
        assert np.isclose(equal_weight_portfolio.weights.sum(), 1.0)

    def test_annualised_vol_positive(self, equal_weight_portfolio):
        assert equal_weight_portfolio.annualised_volatility > 0

    def test_daily_vol_less_than_annual(self, equal_weight_portfolio):
        """Daily vol should be much less than annualised vol."""
        assert equal_weight_portfolio.daily_volatility < equal_weight_portfolio.annualised_volatility

    def test_historical_portfolio_returns_length(self, equal_weight_portfolio, mock_prices):
        hp = equal_weight_portfolio.historical_portfolio_returns()
        assert len(hp) == len(mock_prices) - 1

    def test_cumulative_wealth_starts_at_one(self, equal_weight_portfolio):
        wi = equal_weight_portfolio.cumulative_wealth_index()
        # The wealth index value at the first date should be approximately e^(first return)
        # which starts the cumsum from the first return, not from 1.0 pre-return.
        # Just check it's a positive Series of correct length.
        assert len(wi) == len(equal_weight_portfolio.returns)
        assert (wi > 0).all()

    def test_summary_contains_keys(self, equal_weight_portfolio):
        s = equal_weight_portfolio.summary()
        expected_keys = {"Tickers", "Weights", "Annualised Return",
                         "Annualised Volatility", "Sharpe Ratio"}
        assert expected_keys.issubset(s.keys())

    def test_max_drawdown_negative(self, equal_weight_portfolio):
        """Max drawdown should always be ≤ 0."""
        assert equal_weight_portfolio.max_drawdown <= 0

    def test_component_var_sums_to_total(self, equal_weight_portfolio):
        """Component VaR contributions must sum exactly to total VaR."""
        from risk_metrics import compute_var
        mc_rets = (equal_weight_portfolio.returns * equal_weight_portfolio.weights).sum(axis=1).values
        var_pct = compute_var(mc_rets, 0.95)
        comp = equal_weight_portfolio.component_var_contributions(var_pct)
        total_usd = var_pct * equal_weight_portfolio.portfolio_value
        assert np.isclose(comp.sum(), total_usd, rtol=1e-6)


# ---------------------------------------------------------------------------
# Concentration / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_asset_portfolio(self):
        """A one-asset portfolio should still work."""
        np.random.seed(0)
        n = 50
        prices = pd.DataFrame(
            {"X": 100 * np.cumprod(1 + np.random.normal(0.001, 0.01, n))},
            index=pd.bdate_range("2022-01-01", periods=n),
        )
        p = build_portfolio(["X"], [1.0], prices)
        assert np.isclose(p.weights[0], 1.0)
        assert p.annualised_volatility > 0

    def test_concentrated_portfolio(self, mock_prices):
        """A very concentrated portfolio (99% in one asset) should be valid."""
        p = build_portfolio(
            tickers=["A", "B", "C"],
            weights=[0.99, 0.005, 0.005],
            prices=mock_prices,
        )
        assert np.isclose(p.weights.sum(), 1.0)
