"""
config.py
=========
Central configuration for the Monte Carlo Portfolio Risk Simulator.
Edit this file to change portfolio composition, simulation parameters,
and output settings without touching any core logic.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Portfolio definition
# ---------------------------------------------------------------------------

TICKERS: list[str] = ["SPY", "QQQ", "TLT", "GLD", "AAPL"]

# Weights must sum to 1.0  (validated at runtime)
WEIGHTS: list[float] = [0.30, 0.25, 0.20, 0.15, 0.10]

PORTFOLIO_VALUE: float = 1_000_000.0   # notional value in USD

# ---------------------------------------------------------------------------
# Historical data
# ---------------------------------------------------------------------------

DATA_START: str = "2018-01-01"
DATA_END: str   = "2023-12-31"      # inclusive end date for reproducibility

# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

N_SIMULATIONS: int   = 10_000       # number of simulation paths
RANDOM_SEED:   int   = 42           # seed for reproducibility
TRADING_DAYS:  int   = 252          # trading days per year for annualisation

# Horizons to evaluate (in trading days)
HORIZONS: dict[str, int] = {
    "1-day":   1,
    "10-day":  10,
    "252-day": 252,
}

# Number of full paths to plot on the paths chart
N_PATHS_PLOT: int = 200

# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

CONFIDENCE_LEVELS: list[float] = [0.95, 0.99]

RISK_FREE_RATE: float = 0.04      # annualised risk-free rate (4 %) for Sharpe computation

# ---------------------------------------------------------------------------
# Stress scenarios
# ---------------------------------------------------------------------------

STRESS_SCENARIOS: dict[str, dict] = {
    "Volatility Shock": {
        "description": "Doubles every asset's daily volatility (2σ shock)",
        "vol_multiplier": 2.0,
        "corr_multiplier": 1.0,
        "price_shocks": {},
    },
    "Correlation Shock": {
        "description": "Pushes all pairwise correlations toward +0.85 (crisis regime)",
        "vol_multiplier": 1.5,
        "corr_multiplier": 0.85,   # target correlation floor
        "price_shocks": {},
    },
    "Market Crash": {
        "description": "Equity down -15 %, bond safe-haven up +5 %, gold up +8 %",
        "vol_multiplier": 2.5,
        "corr_multiplier": 0.90,
        "price_shocks": {
            "SPY":  -0.15,
            "QQQ":  -0.20,
            "TLT":   0.05,
            "GLD":   0.08,
            "AAPL": -0.18,
        },
    },
    "Rate Shock": {
        "description": "Sudden 100 bp rate rise; TLT -10 %, equities -5 %",
        "vol_multiplier": 1.8,
        "corr_multiplier": 0.70,
        "price_shocks": {
            "SPY":  -0.05,
            "QQQ":  -0.06,
            "TLT":  -0.10,
            "GLD":  -0.02,
            "AAPL": -0.05,
        },
    },
}

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

OUTPUT_DIR:      str = "outputs"
CHARTS_DIR:      str = "outputs/charts"
TABLES_DIR:      str = "outputs/tables"
SUMMARIES_DIR:   str = "outputs/summaries"

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

FIGURE_DPI: int = 150
FIGURE_SIZE_WIDE: tuple[int, int] = (14, 6)
FIGURE_SIZE_SQUARE: tuple[int, int] = (10, 8)
FIGURE_SIZE_TALL:   tuple[int, int] = (12, 8)
