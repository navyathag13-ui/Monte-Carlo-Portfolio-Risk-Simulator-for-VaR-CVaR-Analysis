"""
utils.py
========
Shared utility helpers used across multiple modules.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dirs(*dirs: str) -> None:
    """Create directories if they do not already exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Weight validation
# ---------------------------------------------------------------------------

def validate_weights(weights: list[float], tickers: list[str]) -> np.ndarray:
    """
    Validate and normalise portfolio weights.

    Parameters
    ----------
    weights : list of float
        Raw weight values.
    tickers : list of str
        Corresponding ticker symbols.

    Returns
    -------
    np.ndarray
        Normalised weight array that sums exactly to 1.0.

    Raises
    ------
    ValueError
        If length mismatch or any weight is negative.
    """
    if len(weights) != len(tickers):
        raise ValueError(
            f"Length mismatch: {len(weights)} weights vs {len(tickers)} tickers."
        )
    w = np.array(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("All weights must be non-negative.")
    total = w.sum()
    if total == 0:
        raise ValueError("Weights must not all be zero.")
    if not np.isclose(total, 1.0, atol=1e-6):
        logger = get_logger(__name__)
        logger.warning(
            "Weights sum to %.6f — normalising to 1.0.", total
        )
        w = w / total
    return w


# ---------------------------------------------------------------------------
# Annualisation helpers
# ---------------------------------------------------------------------------

def annualise_return(daily_return: float, trading_days: int = 252) -> float:
    """Compound a mean daily return to an annualised figure."""
    return (1 + daily_return) ** trading_days - 1


def annualise_volatility(daily_vol: float, trading_days: int = 252) -> float:
    """Scale daily volatility to annualised volatility."""
    return daily_vol * np.sqrt(trading_days)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def save_dataframe(df: pd.DataFrame, path: str, index: bool = True) -> None:
    """Save a DataFrame to CSV, creating parent directories if needed."""
    ensure_dirs(str(Path(path).parent))
    df.to_csv(path, index=index)
    get_logger(__name__).info("Saved table → %s", path)


# ---------------------------------------------------------------------------
# Percentile table
# ---------------------------------------------------------------------------

def build_percentile_table(returns: np.ndarray) -> pd.DataFrame:
    """
    Return a DataFrame of key percentiles for a 1-D return distribution.

    Parameters
    ----------
    returns : np.ndarray
        Simulated portfolio returns (1-D).
    """
    percs = [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99]
    values = np.percentile(returns, percs)
    df = pd.DataFrame({"Percentile (%)": percs, "Portfolio Return": values})
    df["Portfolio Return"] = df["Portfolio Return"].map("{:.4%}".format)
    return df
