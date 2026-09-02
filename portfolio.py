"""
portfolio.py
============
Portfolio object: computes returns, covariance, correlation, and
summary statistics from a cleaned price DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config as _cfg
from utils import (
    get_logger,
    validate_weights,
    annualise_return,
    annualise_volatility,
)

logger = get_logger(__name__)


@dataclass
class Portfolio:
    """
    Represents a multi-asset portfolio with computed risk statistics.

    Attributes
    ----------
    tickers       : asset symbols in order
    weights       : normalised weight array (sums to 1.0)
    prices        : daily adjusted-close price DataFrame
    returns       : daily log-return DataFrame
    mean_returns  : per-asset mean daily log-return
    cov_matrix    : daily covariance matrix
    corr_matrix   : correlation matrix
    portfolio_value : notional value in USD
    """

    tickers:         list[str]
    weights:         np.ndarray
    prices:          pd.DataFrame
    returns:         pd.DataFrame          = field(init=False)
    mean_returns:    pd.Series             = field(init=False)
    cov_matrix:      pd.DataFrame          = field(init=False)
    corr_matrix:     pd.DataFrame          = field(init=False)
    portfolio_value: float                 = 1_000_000.0

    # Post-init computed scalars
    _port_daily_return: float             = field(init=False, repr=False)
    _port_daily_vol:    float             = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._compute_returns()
        self._compute_statistics()

    # ------------------------------------------------------------------
    # Private computations
    # ------------------------------------------------------------------

    def _compute_returns(self) -> None:
        """Compute daily log-returns from adjusted close prices."""
        # Log returns: ln(P_t / P_{t-1})
        self.returns = np.log(self.prices / self.prices.shift(1)).dropna()
        logger.info("Computed %d daily log-return observations.", len(self.returns))

    def _compute_statistics(self) -> None:
        """Compute mean returns, covariance, and correlation matrices."""
        self.mean_returns = self.returns.mean()
        self.cov_matrix   = self.returns.cov()
        self.corr_matrix  = self.returns.corr()

        # Scalar portfolio-level stats
        self._port_daily_return = float(self.weights @ self.mean_returns.values)
        self._port_daily_vol    = float(
            np.sqrt(self.weights @ self.cov_matrix.values @ self.weights)
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def daily_return(self) -> float:
        """Mean daily portfolio log-return."""
        return self._port_daily_return

    @property
    def daily_volatility(self) -> float:
        """Daily portfolio volatility (standard deviation)."""
        return self._port_daily_vol

    @property
    def annualised_return(self) -> float:
        """Annualised portfolio return (compounded, 252 trading days)."""
        return annualise_return(self._port_daily_return)

    @property
    def annualised_volatility(self) -> float:
        """Annualised portfolio volatility."""
        return annualise_volatility(self._port_daily_vol)

    @property
    def sharpe_ratio(self) -> float:
        """
        Annualised Sharpe Ratio using the risk-free rate from config.

        Sharpe = (μ_p - r_f) / σ_p

        Uses config.RISK_FREE_RATE (default 4 %).  A Sharpe above 0.5 is
        considered acceptable for a diversified equity/bond portfolio.
        """
        rf = getattr(_cfg, "RISK_FREE_RATE", 0.04)
        excess = self.annualised_return - rf
        return excess / self.annualised_volatility if self.annualised_volatility else np.nan

    @property
    def max_drawdown(self) -> float:
        """
        Maximum peak-to-trough drawdown of the historical portfolio equity curve.

        Computed on log-returns, so the result is expressed as a negative
        fraction (e.g. -0.34 means a 34 % peak-to-trough loss).  This is a
        widely-used risk metric in portfolio reporting and hedge-fund DD attribution.
        """
        wealth = self.cumulative_wealth_index()
        rolling_peak = wealth.cummax()
        drawdowns = (wealth - rolling_peak) / rolling_peak
        return float(drawdowns.min())

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def historical_portfolio_returns(self) -> pd.Series:
        """
        Daily portfolio-level log-returns using the configured weights.
        Useful for historical simulation VaR.
        """
        return (self.returns * self.weights).sum(axis=1)

    def cumulative_wealth_index(self, initial: float = 1.0) -> pd.Series:
        """
        Cumulative portfolio growth starting from *initial* (default 1.0).
        Uses log-returns so the compounding is exact.
        """
        port_rets = self.historical_portfolio_returns()
        return initial * np.exp(port_rets.cumsum())

    def component_var_contributions(
        self, portfolio_var_pct: float
    ) -> pd.Series:
        """
        Per-asset Component VaR as a fraction of total portfolio VaR.

        Uses the covariance-based decomposition:

            CVaR_i = w_i × (Σw)_i / σ_p  ×  VaR_p

        where (Σw)_i is the i-th element of the covariance-weighted weight
        vector.  Component VaRs sum exactly to the total portfolio VaR,
        making this useful for risk attribution and hedging decisions.

        Parameters
        ----------
        portfolio_var_pct : total portfolio VaR as a positive fraction

        Returns
        -------
        pd.Series  indexed by ticker, values are the $ VaR contribution
        """
        cov   = self.cov_matrix.values
        w     = self.weights
        sigma = self._port_daily_vol                  # portfolio daily vol
        if sigma == 0:
            return pd.Series(np.zeros(len(w)), index=self.tickers)
        # Marginal contribution: ∂σ_p/∂w_i  =  (Σw)_i / σ_p
        marginal = (cov @ w) / sigma
        # Component contribution (fraction of total vol)
        comp_pct = w * marginal / sigma               # sums to 1
        # Scale to VaR dollar contribution
        comp_var_usd = comp_pct * portfolio_var_pct * self.portfolio_value
        return pd.Series(comp_var_usd, index=self.tickers)

    def summary(self) -> dict:
        """Return a dict of key portfolio statistics."""
        return {
            "Tickers":              self.tickers,
            "Weights":              self.weights.tolist(),
            "Daily Return (mean)":  f"{self._port_daily_return:.6f}",
            "Daily Volatility":     f"{self._port_daily_vol:.6f}",
            "Annualised Return":    f"{self.annualised_return:.2%}",
            "Annualised Volatility":f"{self.annualised_volatility:.2%}",
            "Sharpe Ratio":         f"{self.sharpe_ratio:.3f}",
            "Max Drawdown":         f"{self.max_drawdown:.2%}",
            "Observations":         len(self.returns),
        }

    def print_summary(self) -> None:
        """Pretty-print the portfolio summary to stdout."""
        s = self.summary()
        print("\n" + "=" * 60)
        print("  PORTFOLIO SUMMARY")
        print("=" * 60)
        for k, v in s.items():
            label = f"  {k}:"
            print(f"{label:<35} {v}")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_portfolio(
    tickers: list[str],
    weights: list[float],
    prices: pd.DataFrame,
    portfolio_value: float = 1_000_000.0,
) -> Portfolio:
    """
    Convenience constructor that validates inputs before building a Portfolio.

    Parameters
    ----------
    tickers         : list of ticker symbols
    weights         : raw portfolio weights (will be normalised if needed)
    prices          : cleaned price DataFrame from data_loader
    portfolio_value : notional USD portfolio size

    Returns
    -------
    Portfolio instance
    """
    w = validate_weights(weights, tickers)
    # Ensure prices only contains the requested tickers in order
    prices = prices[tickers].copy()
    return Portfolio(
        tickers=tickers,
        weights=w,
        prices=prices,
        portfolio_value=portfolio_value,
    )
