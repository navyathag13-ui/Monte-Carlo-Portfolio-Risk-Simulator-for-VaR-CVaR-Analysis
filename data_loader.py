"""
data_loader.py
==============
Responsible for fetching adjusted-close price data from Yahoo Finance
and falling back to realistic synthetic data when the network is unavailable
or a ticker fails to download.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_price_data(
    tickers: list[str],
    start: str,
    end: str,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Return a DataFrame of daily adjusted-close prices, one column per ticker.

    Strategy
    --------
    1. Attempt to download from Yahoo Finance via yfinance.
    2. For any ticker that fails, substitute synthetic data.
    3. Forward-fill then back-fill any residual NaN values.

    Parameters
    ----------
    tickers  : list of ticker symbols
    start    : ISO date string "YYYY-MM-DD"
    end      : ISO date string "YYYY-MM-DD"
    random_seed : seed for synthetic fallback

    Returns
    -------
    pd.DataFrame with DatetimeIndex and one column per ticker.
    """
    prices = _try_yfinance(tickers, start, end)

    if prices is None or prices.empty:
        logger.warning("Yahoo Finance download failed entirely — using full synthetic data.")
        prices = _synthetic_prices(tickers, start, end, random_seed)
    else:
        # Fill in any missing tickers with synthetic columns
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            logger.warning("Tickers not available from Yahoo Finance: %s", missing)
            synth = _synthetic_prices(missing, start, end, random_seed)
            # Align index
            synth = synth.reindex(prices.index).ffill().bfill()
            prices = pd.concat([prices, synth], axis=1)

    # Final clean-up
    prices = prices[tickers]          # enforce original ticker order
    prices = prices.ffill().bfill()   # handle any residual NaNs
    prices = prices.dropna()

    logger.info(
        "Price data loaded: %d trading days × %d tickers  [%s → %s]",
        len(prices), len(prices.columns),
        prices.index[0].date(), prices.index[-1].date(),
    )
    return prices


# ---------------------------------------------------------------------------
# Yahoo Finance download
# ---------------------------------------------------------------------------

def _try_yfinance(
    tickers: list[str], start: str, end: str
) -> pd.DataFrame | None:
    """Attempt a yfinance download; return None on any failure."""
    try:
        import yfinance as yf  # lazy import — optional dependency

        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if raw.empty:
            return None

        # yfinance returns multi-level columns when >1 ticker is requested
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                prices = raw["Close"]
            else:
                return None
        else:
            # Single ticker — wrap in DataFrame
            prices = raw[["Close"]] if "Close" in raw.columns else raw
            prices.columns = tickers[:1]

        # Drop columns that are entirely NaN
        prices = prices.dropna(axis=1, how="all")
        return prices

    except Exception as exc:
        logger.warning("yfinance download raised an exception: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Synthetic price generator
# ---------------------------------------------------------------------------

# Realistic long-run parameters for the default tickers
_TICKER_PARAMS: dict[str, dict] = {
    "SPY":  {"mu": 0.10, "vol": 0.16, "s0": 380.0},
    "QQQ":  {"mu": 0.14, "vol": 0.21, "s0": 310.0},
    "TLT":  {"mu": 0.02, "vol": 0.14, "s0": 140.0},
    "GLD":  {"mu": 0.05, "vol": 0.13, "s0": 170.0},
    "AAPL": {"mu": 0.20, "vol": 0.28, "s0": 150.0},
}
_DEFAULT_PARAMS: dict = {"mu": 0.08, "vol": 0.20, "s0": 100.0}


def _synthetic_prices(
    tickers: list[str],
    start: str,
    end: str,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate correlated GBM price paths for the supplied tickers.

    Uses a mild positive correlation structure that mimics a diversified
    equity / bond / commodity portfolio (equities are correlated; bonds
    and gold show negative or low correlation to equities).
    """
    rng = np.random.default_rng(random_seed)

    # Build a business-day date index
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # Asset-specific parameters
    params = [_TICKER_PARAMS.get(t, _DEFAULT_PARAMS) for t in tickers]
    dt = 1 / 252

    # Correlation matrix — realistic rough approximation
    k = len(tickers)
    corr = _build_default_corr(tickers, k)

    # Cholesky decomposition for correlated normals
    L = np.linalg.cholesky(corr)

    # Simulate log-returns
    z = rng.standard_normal((n, k))
    z_corr = z @ L.T

    log_returns = np.array([
        (p["mu"] - 0.5 * p["vol"] ** 2) * dt + p["vol"] * np.sqrt(dt) * z_corr[:, i]
        for i, p in enumerate(params)
    ]).T  # shape (n, k)

    # Cumulative prices
    s0 = np.array([p["s0"] for p in params])
    prices = s0 * np.exp(np.cumsum(log_returns, axis=0))

    df = pd.DataFrame(prices, index=dates, columns=tickers)
    logger.info("Synthetic price data generated for: %s", tickers)
    return df


def _build_default_corr(tickers: list[str], k: int) -> np.ndarray:
    """
    Build a plausible correlation matrix for a mixed equity/bond/commodity
    portfolio.  Falls back to a mild uniform correlation when tickers are unknown.
    """
    # Pair-wise correlation lookup (equity-equity ~0.75, equity-bond ~-0.10, etc.)
    CORR_LOOKUP: dict[tuple[str, str], float] = {
        ("SPY", "QQQ"):  0.88,
        ("SPY", "TLT"):  -0.05,
        ("SPY", "GLD"):   0.03,
        ("SPY", "AAPL"):  0.75,
        ("QQQ", "TLT"):  -0.10,
        ("QQQ", "GLD"):   0.00,
        ("QQQ", "AAPL"):  0.82,
        ("TLT", "GLD"):   0.15,
        ("TLT", "AAPL"): -0.08,
        ("GLD", "AAPL"):  0.02,
    }

    corr = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            key1 = (tickers[i], tickers[j])
            key2 = (tickers[j], tickers[i])
            val = CORR_LOOKUP.get(key1, CORR_LOOKUP.get(key2, 0.30))
            corr[i, j] = val
            corr[j, i] = val

    # Ensure positive-definite (small regularisation)
    eigvals = np.linalg.eigvalsh(corr)
    if eigvals.min() < 1e-6:
        corr += (1e-4 - eigvals.min()) * np.eye(k)

    return corr
