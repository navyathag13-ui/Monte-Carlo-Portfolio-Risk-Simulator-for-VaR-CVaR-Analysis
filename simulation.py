"""
simulation.py
=============
Monte Carlo simulation engine for portfolio returns.

Supports two simulation modes:
  1. Single-period portfolio loss distribution (used for VaR/CVaR).
  2. Multi-step cumulative path simulation (used for path visualisation).

Both modes use a multivariate normal framework with a covariance-aware
Cholesky decomposition for proper cross-asset correlation structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio import Portfolio
from utils import get_logger

logger = get_logger(__name__)


@dataclass
class SimulationResult:
    """
    Stores the outputs of a Monte Carlo simulation run.

    Attributes
    ----------
    horizon          : simulation horizon in trading days
    n_simulations    : number of paths simulated
    portfolio_returns: 1-D array of simulated horizon portfolio log-returns
    paths            : 2-D array (horizon × n_simulations) of per-step returns
                       (None for single-period mode)
    cumulative_paths : 2-D array (horizon+1 × n_simulations) of wealth index
                       starting at 1.0  (None for single-period mode)
    """

    horizon:           int
    n_simulations:     int
    portfolio_returns: np.ndarray
    paths:             np.ndarray | None = None
    cumulative_paths:  np.ndarray | None = None


# ---------------------------------------------------------------------------
# Core simulation function
# ---------------------------------------------------------------------------

def run_monte_carlo(
    portfolio: Portfolio,
    horizon: int,
    n_simulations: int = 10_000,
    random_seed: int = 42,
    full_paths: bool = False,
    cov_override: np.ndarray | None = None,
) -> SimulationResult:
    """
    Run a Monte Carlo simulation of portfolio returns over *horizon* trading
    days using a multivariate normal (Cholesky) framework.

    Parameters
    ----------
    portfolio     : fitted Portfolio object
    horizon       : number of trading days to simulate
    n_simulations : number of independent sample paths
    random_seed   : NumPy random seed for reproducibility
    full_paths    : if True, store every daily step (needed for path plots)
    cov_override  : optional alternative covariance matrix (for stress tests)

    Returns
    -------
    SimulationResult
    """
    rng = np.random.default_rng(random_seed)

    mu  = portfolio.mean_returns.values            # shape (k,)
    cov = cov_override if cov_override is not None else portfolio.cov_matrix.values
    w   = portfolio.weights                         # shape (k,)
    k   = len(mu)

    # Cholesky decomposition for correlated normals
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Regularise if matrix is not positive definite
        logger.warning("Covariance matrix not PD — applying regularisation.")
        cov_reg = cov + 1e-8 * np.eye(k)
        L = np.linalg.cholesky(cov_reg)

    if horizon == 1:
        # ------------------------------------------------------------------
        # Single-period simulation (fast path)
        # ------------------------------------------------------------------
        z = rng.standard_normal((n_simulations, k))
        asset_returns = mu + (z @ L.T)             # shape (n_simulations, k)
        port_returns  = asset_returns @ w           # shape (n_simulations,)

        return SimulationResult(
            horizon=horizon,
            n_simulations=n_simulations,
            portfolio_returns=port_returns,
        )

    else:
        # ------------------------------------------------------------------
        # Multi-step simulation
        # ------------------------------------------------------------------
        # Shape: (horizon, n_simulations, k)
        z = rng.standard_normal((horizon, n_simulations, k))

        # Broadcast mu and apply Cholesky: each step is correlated
        step_returns = mu[np.newaxis, np.newaxis, :] + (z @ L.T)
        # step_returns shape: (horizon, n_simulations, k)

        # Portfolio daily returns per step: (horizon, n_simulations)
        port_step_returns = step_returns @ w

        # Cumulative log-return over full horizon
        port_cumulative = port_step_returns.sum(axis=0)  # (n_simulations,)

        result = SimulationResult(
            horizon=horizon,
            n_simulations=n_simulations,
            portfolio_returns=port_cumulative,
        )

        if full_paths:
            result.paths = port_step_returns          # (horizon, n_simulations)
            # Wealth index starting at 1.0
            cum = np.exp(np.cumsum(port_step_returns, axis=0))
            ones = np.ones((1, n_simulations))
            result.cumulative_paths = np.vstack([ones, cum])  # (horizon+1, n_simulations)

        return result


# ---------------------------------------------------------------------------
# Convenience: run all configured horizons
# ---------------------------------------------------------------------------

def run_all_horizons(
    portfolio: Portfolio,
    horizons: dict[str, int],
    n_simulations: int = 10_000,
    random_seed: int = 42,
    full_paths_horizon: str = "252-day",
) -> dict[str, SimulationResult]:
    """
    Run simulations for every horizon in *horizons*.

    The horizon labelled *full_paths_horizon* will store cumulative paths
    for visualisation; others use the fast single-period or aggregate mode.

    Parameters
    ----------
    portfolio          : fitted Portfolio object
    horizons           : mapping of label → trading-days count
    n_simulations      : number of paths per horizon
    random_seed        : seed (incremented per horizon for independence)
    full_paths_horizon : key in *horizons* for which to store full paths

    Returns
    -------
    dict mapping horizon label → SimulationResult
    """
    results: dict[str, SimulationResult] = {}

    for i, (label, h) in enumerate(horizons.items()):
        need_paths = (label == full_paths_horizon)
        logger.info(
            "Simulating horizon: %-10s  (%d steps, %d paths, full_paths=%s)",
            label, h, n_simulations, need_paths,
        )
        results[label] = run_monte_carlo(
            portfolio=portfolio,
            horizon=h,
            n_simulations=n_simulations,
            random_seed=random_seed + i,   # different seed per horizon
            full_paths=need_paths,
        )

    return results
