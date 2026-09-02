"""
stress_testing.py
=================
Applies stress scenarios to the portfolio covariance / mean-return
parameters and re-runs the Monte Carlo simulation under each scenario,
producing a comparison table of baseline vs stressed risk metrics.
"""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd

from portfolio import Portfolio
from simulation import run_monte_carlo, SimulationResult
from risk_metrics import compute_var, compute_cvar
from utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Stress covariance builder
# ---------------------------------------------------------------------------

def _stressed_covariance(
    cov: np.ndarray,
    vol_multiplier: float = 1.0,
    corr_target: float | None = None,
) -> np.ndarray:
    """
    Return a stressed covariance matrix.

    Parameters
    ----------
    cov            : baseline daily covariance matrix (k × k)
    vol_multiplier : multiply every diagonal element (variance) by this factor
    corr_target    : if provided, push all off-diagonal correlations toward
                     this value (crisis correlation shock).  Values between
                     original and target are blended using the larger.

    Returns
    -------
    Stressed covariance matrix.
    """
    # Extract standard deviations and correlation matrix
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)

    # Scale standard deviations
    stressed_std = std * np.sqrt(vol_multiplier)

    # Optionally raise correlations toward target
    if corr_target is not None:
        k = corr.shape[0]
        for i in range(k):
            for j in range(i + 1, k):
                new_val = max(corr[i, j], corr_target)
                new_val = min(new_val, 0.999)   # cap below 1.0
                corr[i, j] = new_val
                corr[j, i] = new_val

    # Rebuild covariance: Σ = D · Corr · D  where D = diag(σ)
    stressed_cov = np.outer(stressed_std, stressed_std) * corr
    np.fill_diagonal(stressed_cov, stressed_std ** 2)

    # Ensure positive-definiteness
    eigvals = np.linalg.eigvalsh(stressed_cov)
    if eigvals.min() < 1e-10:
        stressed_cov += (abs(eigvals.min()) + 1e-8) * np.eye(len(stressed_std))

    return stressed_cov


def _stressed_mean_returns(
    mean_returns: np.ndarray,
    tickers: list[str],
    price_shocks: dict[str, float],
) -> np.ndarray:
    """
    Apply instantaneous price shocks by adjusting mean log-returns.

    A price shock of -15 % is converted to a log-return adjustment of
    ln(1 - 0.15), added once to the mean.  For a 1-day horizon this
    approximates an immediate gap move.
    """
    adjusted = mean_returns.copy()
    for i, ticker in enumerate(tickers):
        if ticker in price_shocks:
            shock_pct = price_shocks[ticker]
            adjusted[i] += np.log(1 + shock_pct)
    return adjusted


# ---------------------------------------------------------------------------
# Single scenario runner
# ---------------------------------------------------------------------------

def run_stress_scenario(
    portfolio: Portfolio,
    scenario: dict,
    horizon: int = 1,
    n_simulations: int = 10_000,
    random_seed: int = 42,
) -> SimulationResult:
    """
    Run the Monte Carlo simulation under a stress scenario.

    Parameters
    ----------
    portfolio     : fitted Portfolio object
    scenario      : scenario definition dict from config.STRESS_SCENARIOS
    horizon       : simulation horizon in trading days
    n_simulations : number of paths
    random_seed   : seed

    Returns
    -------
    SimulationResult with stressed portfolio returns
    """
    cov_base = portfolio.cov_matrix.values
    mu_base  = portfolio.mean_returns.values.copy()

    # Build stressed covariance
    vol_mult   = scenario.get("vol_multiplier", 1.0)
    corr_mult  = scenario.get("corr_multiplier", None)
    if isinstance(corr_mult, (int, float)) and corr_mult == 1.0:
        corr_mult = None   # no correlation shock

    cov_stressed = _stressed_covariance(
        cov_base,
        vol_multiplier=vol_mult,
        corr_target=corr_mult,
    )

    # Apply price shocks to mean returns
    price_shocks = scenario.get("price_shocks", {})
    mu_stressed = _stressed_mean_returns(mu_base, portfolio.tickers, price_shocks)

    # Deep-copy portfolio so we don't mutate the original object
    stressed_port = copy.deepcopy(portfolio)
    stressed_port.mean_returns = pd.Series(mu_stressed, index=portfolio.mean_returns.index)

    result = run_monte_carlo(
        portfolio=stressed_port,
        horizon=horizon,
        n_simulations=n_simulations,
        random_seed=random_seed,
        cov_override=cov_stressed,
    )
    return result


# ---------------------------------------------------------------------------
# Full scenario comparison table
# ---------------------------------------------------------------------------

def run_all_stress_scenarios(
    portfolio: Portfolio,
    scenarios: dict[str, dict],
    confidence_levels: list[float],
    horizon: int = 1,
    n_simulations: int = 10_000,
    random_seed: int = 42,
    portfolio_value: float = 1_000_000.0,
) -> pd.DataFrame:
    """
    Run all stress scenarios and return a comparison DataFrame.

    Parameters
    ----------
    portfolio          : fitted Portfolio object
    scenarios          : dict of scenario label → scenario config dict
    confidence_levels  : e.g. [0.95, 0.99]
    horizon            : simulation horizon (days)
    n_simulations      : paths per scenario
    random_seed        : seed
    portfolio_value    : notional value for dollar conversion

    Returns
    -------
    DataFrame with columns: Scenario, Confidence, VaR (%), CVaR (%),
                             VaR ($), CVaR ($), VaR Change, CVaR Change
    """
    rows = []

    # Baseline first
    baseline_result = run_monte_carlo(
        portfolio=portfolio,
        horizon=horizon,
        n_simulations=n_simulations,
        random_seed=random_seed,
    )
    baseline_rets = baseline_result.portfolio_returns

    for cl in confidence_levels:
        base_var  = compute_var(baseline_rets,  cl)
        base_cvar = compute_cvar(baseline_rets, cl)
        rows.append({
            "Scenario":    "Baseline",
            "Confidence":  f"{cl:.0%}",
            "VaR (%)":     f"{base_var:.4%}",
            "CVaR (%)":    f"{base_cvar:.4%}",
            "VaR ($)":     f"${base_var  * portfolio_value:,.0f}",
            "CVaR ($)":    f"${base_cvar * portfolio_value:,.0f}",
            "VaR Change":  "—",
            "CVaR Change": "—",
        })

    # Stressed scenarios
    for name, config in scenarios.items():
        logger.info("Running stress scenario: %s", name)
        result = run_stress_scenario(
            portfolio=portfolio,
            scenario=config,
            horizon=horizon,
            n_simulations=n_simulations,
            random_seed=random_seed,
        )
        rets = result.portfolio_returns
        for cl in confidence_levels:
            base_var  = compute_var(baseline_rets,  cl)
            base_cvar = compute_cvar(baseline_rets, cl)
            s_var     = compute_var(rets,  cl)
            s_cvar    = compute_cvar(rets, cl)
            rows.append({
                "Scenario":    name,
                "Confidence":  f"{cl:.0%}",
                "VaR (%)":     f"{s_var:.4%}",
                "CVaR (%)":    f"{s_cvar:.4%}",
                "VaR ($)":     f"${s_var  * portfolio_value:,.0f}",
                "CVaR ($)":    f"${s_cvar * portfolio_value:,.0f}",
                "VaR Change":  f"{(s_var  - base_var)  / base_var:+.1%}",
                "CVaR Change": f"{(s_cvar - base_cvar) / base_cvar:+.1%}",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Identify worst scenario
# ---------------------------------------------------------------------------

def worst_scenario(
    scenario_df: pd.DataFrame,
    confidence: str = "95%",
    metric: str = "CVaR (%)",
) -> str:
    """
    Return the name of the scenario with the highest CVaR at *confidence*.

    Parses percentage strings back to floats for comparison.
    """
    sub = scenario_df[
        (scenario_df["Confidence"] == confidence) &
        (scenario_df["Scenario"] != "Baseline")
    ].copy()

    def _parse_pct(s: str) -> float:
        return float(str(s).replace("%", "").strip()) / 100

    sub["_metric"] = sub[metric].apply(_parse_pct)
    idx = sub["_metric"].idxmax()
    return str(sub.loc[idx, "Scenario"])
