"""
risk_metrics.py
===============
VaR, CVaR, and supplementary risk measures computed from a simulated
return distribution.

Three estimation methods are provided for comparison:
  1. Monte Carlo (simulation-based)
  2. Historical Simulation
  3. Parametric / Variance-Covariance (Gaussian)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core VaR / CVaR functions
# ---------------------------------------------------------------------------

def compute_var(
    returns: np.ndarray,
    confidence: float,
    method: str = "monte_carlo",
) -> float:
    """
    Value at Risk at a given confidence level.

    VaR represents the loss that will **not** be exceeded with probability
    equal to *confidence*.  Returned as a **positive number** (loss magnitude).

    Parameters
    ----------
    returns    : 1-D array of simulated or historical portfolio log-returns.
    confidence : confidence level, e.g. 0.95 or 0.99.
    method     : "monte_carlo" (empirical percentile) or "parametric".

    Returns
    -------
    float
        VaR as a positive loss magnitude (e.g. 0.025 means 2.5 % loss).
    """
    if method == "parametric":
        mu  = float(np.mean(returns))
        sig = float(np.std(returns, ddof=1))
        z   = stats.norm.ppf(1 - confidence)
        return float(-(mu + z * sig))
    else:
        # Empirical (simulation / historical)
        return float(-np.percentile(returns, (1 - confidence) * 100))


def compute_cvar(
    returns: np.ndarray,
    confidence: float,
    method: str = "monte_carlo",
) -> float:
    """
    Conditional Value at Risk (Expected Shortfall) at a given confidence level.

    CVaR is the **expected loss** given that losses exceed VaR.
    Returned as a **positive number**.

    Parameters
    ----------
    returns    : 1-D array of portfolio log-returns.
    confidence : confidence level.
    method     : "monte_carlo" (empirical tail mean) or "parametric".

    Returns
    -------
    float
        CVaR as a positive loss magnitude.
    """
    if method == "parametric":
        mu  = float(np.mean(returns))
        sig = float(np.std(returns, ddof=1))
        z   = stats.norm.ppf(1 - confidence)
        pdf_z = stats.norm.pdf(z)
        return float(-(mu - sig * pdf_z / (1 - confidence)))
    else:
        var = compute_var(returns, confidence, method="monte_carlo")
        tail = returns[returns <= -var]
        if len(tail) == 0:
            return var          # fallback — shouldn't happen with large N
        return float(-tail.mean())


# ---------------------------------------------------------------------------
# Multi-level risk table
# ---------------------------------------------------------------------------

def build_risk_table(
    mc_returns: np.ndarray,
    hist_returns: np.ndarray | None,
    confidence_levels: list[float],
    portfolio_value: float = 1_000_000.0,
) -> pd.DataFrame:
    """
    Build a comprehensive risk metrics table comparing Monte Carlo,
    Historical Simulation, and Parametric methods.

    Parameters
    ----------
    mc_returns         : simulated portfolio log-returns (1-D)
    hist_returns       : historical portfolio log-returns (1-D), may be None
    confidence_levels  : list of confidence levels, e.g. [0.95, 0.99]
    portfolio_value    : notional value to convert % losses to dollar amounts

    Returns
    -------
    pd.DataFrame with columns: Method, Confidence, VaR (%), VaR ($),
                                CVaR (%), CVaR ($)
    """
    rows = []

    method_data = {
        "Monte Carlo":  (mc_returns,   "monte_carlo"),
        "Parametric":   (mc_returns,   "parametric"),
    }
    if hist_returns is not None and len(hist_returns) > 0:
        method_data["Historical"] = (hist_returns, "monte_carlo")

    for method_label, (rets, method_key) in method_data.items():
        for cl in confidence_levels:
            var  = compute_var(rets, cl, method=method_key)
            cvar = compute_cvar(rets, cl, method=method_key)
            rows.append({
                "Method":         method_label,
                "Confidence":     f"{cl:.0%}",
                "VaR (%)":        f"{var:.4%}",
                "VaR ($)":        f"${var * portfolio_value:,.0f}",
                "CVaR (%)":       f"{cvar:.4%}",
                "CVaR ($)":       f"${cvar * portfolio_value:,.0f}",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Single-confidence result (used by stress testing and summary)
# ---------------------------------------------------------------------------

def risk_at_confidence(
    returns: np.ndarray,
    confidence: float,
    portfolio_value: float = 1_000_000.0,
) -> dict[str, float]:
    """
    Compute VaR and CVaR for a single confidence level and return raw floats.

    Returns
    -------
    dict with keys: var_pct, cvar_pct, var_usd, cvar_usd
    """
    var  = compute_var(returns, confidence)
    cvar = compute_cvar(returns, confidence)
    return {
        "var_pct":  var,
        "cvar_pct": cvar,
        "var_usd":  var  * portfolio_value,
        "cvar_usd": cvar * portfolio_value,
    }


# ---------------------------------------------------------------------------
# Distribution statistics
# ---------------------------------------------------------------------------

def distribution_stats(returns: np.ndarray) -> dict[str, float]:
    """
    Descriptive statistics for the simulated return distribution.

    Returns
    -------
    dict with mean, std, skewness, kurtosis (excess), min, max, and
    various percentiles.
    """
    return {
        "mean":      float(np.mean(returns)),
        "std":       float(np.std(returns, ddof=1)),
        "skewness":  float(stats.skew(returns)),
        "kurtosis":  float(stats.kurtosis(returns)),   # excess kurtosis
        "min":       float(np.min(returns)),
        "max":       float(np.max(returns)),
        "p1":        float(np.percentile(returns, 1)),
        "p5":        float(np.percentile(returns, 5)),
        "p25":       float(np.percentile(returns, 25)),
        "median":    float(np.median(returns)),
        "p75":       float(np.percentile(returns, 75)),
        "p95":       float(np.percentile(returns, 95)),
        "p99":       float(np.percentile(returns, 99)),
    }
