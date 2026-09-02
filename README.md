# Monte Carlo Portfolio Risk Simulator
### VaR · CVaR · Stress Testing · Risk Attribution

> A production-style Python toolkit for quantitative portfolio risk analysis — end-to-end from raw price data to institutional-grade risk reports, stress scenarios, and a live interactive dashboard.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-50%20passed-27ae60?logo=pytest&logoColor=white)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-f39c12)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](#interactive-dashboard)

---

## Overview

Risk quantification is at the centre of modern finance. Whether a derivatives desk is sizing hedges, a prime broker is computing margin, or a portfolio manager is allocating capital under a VaR budget, the underlying workflow is the same: simulate future losses, characterise their distribution, and stress-test the tail.

This project implements that workflow in full — from covariance estimation and Cholesky-decomposed Monte Carlo simulation through to VaR/CVaR calculation, multi-scenario stress testing, component risk attribution, and a Streamlit dashboard — all wired together with clean, modular, tested Python.

Built as a portfolio project targeting roles in **quantitative risk, derivatives analytics, market risk, and financial engineering**.

---

## What Makes This Interview-Ready

| Dimension | What's here |
|---|---|
| **Quant Rigour** | Cholesky multivariate normal simulation; coherent CVaR (Expected Shortfall); three VaR methods cross-validated |
| **Regulatory Framing** | Basel II 10-day 99% VaR; FRTB/Basel IV 97.5% ES; DFAST/CCAR-style scenario design |
| **Risk Attribution** | Component VaR decomposition — marginal contribution per asset, sums exactly to total |
| **Stress Testing** | 4 calibrated scenarios: volatility shock, correlation breakdown, market crash, rate shock |
| **Engineering** | 8-module clean architecture; 50 unit tests; CLI; Streamlit dashboard; reproducible seed |
| **Reporting** | 9 publication-quality charts; 6 CSVs; auto-generated Markdown risk report |

---

## Live Results (Default Portfolio)

**Portfolio:** SPY 30% · QQQ 25% · TLT 20% · GLD 15% · AAPL 10%
**Notional:** $1,000,000 · **Horizon:** 2018–2023 · **Simulations:** 10,000

| Metric | Value |
|---|---|
| Annualised Return | **11.43%** |
| Annualised Volatility | **15.29%** |
| Sharpe Ratio (r_f = 4%) | **0.486** |
| Max Historical Drawdown | **−26.77%** |
| **95% VaR (1-day)** | **1.56% · $15,616** |
| **99% VaR (1-day)** | **2.16% · $21,553** |
| **95% CVaR / ES (1-day)** | **1.94% · $19,424** |
| **99% CVaR / ES (1-day)** | **2.47% · $24,721** |
| Worst Stress Scenario | **Market Crash (+231% CVaR vs baseline)** |

### Component VaR Attribution (95%)
```
QQQ   $6,077  38.9%  ████████████
SPY   $5,769  36.9%  ███████████
AAPL  $2,784  17.8%  █████
GLD     $656   4.2%  █
TLT     $329   2.1%  █
```
> QQQ and SPY together drive **75.8%** of total risk despite representing 55% of the portfolio — the classic concentration vs. diversification tension.

---

## Features

```
✓  Multi-asset portfolio with configurable weights + notional
✓  Yahoo Finance data download with realistic synthetic fallback
✓  Monte Carlo simulation — 10,000 paths, Cholesky multivariate normal
✓  Three VaR methods: Monte Carlo · Historical Simulation · Parametric
✓  VaR and CVaR at 95% and 99% across 1-day, 10-day, 252-day horizons
✓  Component VaR attribution — per-asset marginal risk contribution
✓  Max drawdown from historical equity curve
✓  4 stress scenarios: vol shock, correlation shock, market crash, rate shock
✓  Baseline vs stressed VaR/CVaR comparison table with % change
✓  9 publication-quality charts saved to outputs/charts/
✓  6 structured CSVs + auto-generated Markdown risk report
✓  Streamlit dashboard with live inputs, tabs, and CSV download
✓  CLI interface — override tickers, weights, sims, dates from terminal
✓  50-test pytest suite covering all core modules
```

---

## Methodology

### 1 · Return Estimation

Daily log-returns are computed as:

```
r_t = ln(P_t / P_{t-1})
```

The portfolio-level daily return is the weight-dot return:

```
μ_p = wᵀ · μ        σ_p = √(wᵀ Σ w)
```

Annualisation uses 252 trading days:

```
μ_annual = (1 + μ_daily)^252 − 1
σ_annual = σ_daily × √252
```

### 2 · Monte Carlo Simulation

Returns are drawn from a **multivariate normal** with covariance structure preserved via the **Cholesky decomposition**:

```
r_t = μ + L · z_t     where  z_t ~ N(0, I)  and  L = chol(Σ)
```

This ensures simulated asset returns respect the observed cross-asset correlation matrix — a critical property absent in naive per-asset independent sampling.

For multi-step horizons, steps compound via:

```
wealth_T = exp( Σ_t  r_t )     t = 1…T
```

### 3 · Value at Risk (VaR)

VaR is the **maximum loss not exceeded at a given confidence level** over the horizon:

```
VaR_α = −inf{ l : P(L ≤ l) ≥ α }
```

In plain terms: *"There is a 5% chance the portfolio loses more than X% in a single day."*

Three estimation methods are computed and cross-validated:

| Method | Approach |
|---|---|
| **Monte Carlo** | Empirical percentile of the simulated loss distribution |
| **Historical Simulation** | Empirical percentile of realised daily returns |
| **Parametric** | Closed-form Gaussian: VaR = −(μ + z_α · σ) |

### 4 · Conditional VaR / Expected Shortfall (CVaR)

CVaR answers the harder question: *"Given that we are in the tail, how bad is it expected to be?"*

```
CVaR_α = E[ L | L > VaR_α ]
```

CVaR is a **coherent risk measure** — it satisfies sub-additivity, meaning the CVaR of a combined portfolio is no greater than the sum of individual CVaRs. This property makes it the preferred metric under **Basel III/IV FRTB**, where the 97.5% Expected Shortfall has replaced the 99% VaR for internal models.

> CVaR is always ≥ VaR at the same confidence level. The ratio CVaR/VaR captures tail severity — for normal returns it is approximately 1.25; for heavy-tailed or skewed distributions it can exceed 1.5.

### 5 · Component VaR Attribution

Total portfolio VaR is decomposed into per-asset contributions using the covariance-based marginal approach:

```
ComponentVaR_i = w_i · (Σw)_i / σ_p  ×  VaR_p
```

Component VaRs sum exactly to the total portfolio VaR, making them directly useful for hedging overlay design: the largest contributor is the primary candidate for a risk-reducing hedge.

### 6 · Stress Testing

Each stress scenario applies calibrated shocks to the covariance matrix and/or mean returns, then re-runs the full Monte Carlo simulation:

| Scenario | Design |
|---|---|
| **Volatility Shock** | All asset vols ×2 — simulates a VIX spike to ~60 |
| **Correlation Shock** | All pairwise correlations floored at 0.85 — crisis regime where diversification collapses |
| **Market Crash** | SPY −15%, QQQ −20%, TLT +5%, GLD +8%, AAPL −18% + 2.5× vol — 2008/2020-style event |
| **Rate Shock** | TLT −10%, equities −5%, 1.8× vol — instantaneous +100bp parallel rate rise |

The methodology mirrors DFAST (Dodd-Frank), CCAR, and EBA stress-testing frameworks used by major banks.

---

## Project Architecture

```
Monte-Carlo-Portfolio-Risk-Simulator/
│
├── config.py            ← Single source of truth: tickers, weights, scenarios, seeds
├── data_loader.py       ← yfinance download + correlated GBM synthetic fallback
├── portfolio.py         ← Portfolio dataclass: returns, cov, corr, VaR attribution
├── simulation.py        ← Cholesky Monte Carlo engine, multi-step path generator
├── risk_metrics.py      ← VaR, CVaR, three methods, percentile tables
├── stress_testing.py    ← Covariance stressing, price shocks, scenario comparison
├── visualization.py     ← 9 chart generators (matplotlib only, no seaborn)
├── utils.py             ← Logging, weight validation, annualisation, CSV helpers
├── main.py              ← Orchestrator: CLI + full workflow + interpretation report
├── app.py               ← Streamlit interactive dashboard
│
├── tests/
│   ├── test_risk_metrics.py    17 tests — VaR/CVaR correctness, edge cases
│   ├── test_portfolio.py       19 tests — construction, attribution, drawdown
│   └── test_simulation.py      14 tests — engine correctness, stress scenarios
│
├── outputs/
│   ├── charts/          ← 9 PNG figures
│   ├── tables/          ← 6 CSV files
│   └── summaries/       ← Markdown risk report
│
├── notebooks/
│   └── exploration.ipynb   ← Step-by-step methodology walkthrough
│
├── requirements.txt
├── .gitignore
└── README.md
```

**Design principles:** single-responsibility modules; no circular imports; every public function has a docstring with parameters, returns, and financial context; config is the only file a user needs to touch to change the portfolio.

---

## Generated Charts

| # | Filename | What It Shows |
|---|---|---|
| 01 | `01_return_distribution_1d.png` | Simulated 1-day return histogram with VaR/CVaR lines and shaded ES zone |
| 02 | `02_pnl_distribution_1d.png` | Dollar P&L split into gain/loss with VaR markers |
| 03 | `03_correlation_heatmap.png` | Pairwise asset correlation — diversification diagnostic |
| 04 | `04_cumulative_returns.png` | Historical growth curves per asset + blended portfolio |
| 05 | `05_simulation_paths_252d.png` | 200 sample MC paths + 1st/5th/25th–75th/95th percentile fan |
| 06 | `06_stress_comparison_95.png` | Grouped bar: baseline vs all 4 stress scenarios |
| 07 | `07_multi_horizon_var.png` | VaR in USD across 1-day, 10-day, 252-day with √T reference |
| 08 | `08_return_distribution_10d.png` | 10-day distribution (Basel II capital charge horizon) |
| 09 | `09_component_var_attribution.png` | Per-asset VaR contribution — primary hedging signal |

---

## Installation

```bash
# Clone
git clone https://github.com/navyathag13-ui/Monte-Carlo-Portfolio-Risk-Simulator-for-VaR-CVaR-Analysis.git
cd Monte-Carlo-Portfolio-Risk-Simulator-for-VaR-CVaR-Analysis

# Virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Dependencies
pip install -r requirements.txt
```

---

## How to Run

### Command-line simulator
```bash
# Default portfolio (SPY QQQ TLT GLD AAPL)
python main.py

# Custom tickers and weights
python main.py --tickers SPY QQQ GLD --weights 0.50 0.30 0.20

# Larger book, more simulations
python main.py --value 5000000 --sims 50000

# Skip chart generation (fast mode for CI)
python main.py --no-plots
```

### Interactive Streamlit dashboard
```bash
streamlit run app.py
# Open http://localhost:8501
```

The dashboard supports live portfolio reconfiguration, scenario exploration, and one-click CSV export — no coding required.

### Run tests
```bash
pytest tests/ -v
# 50 passed in ~1.5s
```

---

## Interpreting the Output

### VaR — the headline number
> *"There is a 5% probability that the portfolio will lose more than **$15,616** on any given trading day."*

VaR tells you the **frequency** of large losses. It answers: "how often does something bad happen?"

### CVaR — the severity number
> *"On the days when losses exceed the 95% VaR threshold, the average loss is **$19,424**."*

CVaR tells you the **magnitude** of losses in the tail. It answers: "when something bad happens, how bad is it?" This is why regulators (FRTB/Basel IV) prefer CVaR — a portfolio can have the same VaR as another but dramatically different tail severity.

### √T scaling
The 10-day VaR is approximately $15,616 × √10 ≈ $49,400, consistent with the Basel II capital charge convention. Chart 07 shows this scaling explicitly.

### Stress scenarios
The Market Crash scenario produces a **231% increase in CVaR** vs baseline — demonstrating that the portfolio's apparent diversification benefits collapse under a correlated equity sell-off, as they did in March 2020 and Q4 2008.

### Component VaR
QQQ and SPY together account for **75.8% of total VaR** despite representing 55% of the portfolio. This concentration signal directly informs a practical hedging question: buying SPX puts or a QQQ inverse ETF would address the majority of the portfolio's risk budget.

---

## Limitations & Assumptions

| Assumption | Implication |
|---|---|
| Multivariate normal returns | Underestimates fat tails and left skew present in actual equity returns |
| Static covariance matrix | Volatility and correlation are time-varying; a GARCH/DCC overlay would improve accuracy |
| Log-normal price dynamics | Ignores jumps, mean-reversion, and market microstructure |
| Full liquidity | Does not model market impact or bid-ask spread for large positions |
| i.i.d. returns | No autocorrelation or volatility clustering (ARCH effects) modelled |
| 252-day annualisation | Standard convention; actual trading calendar varies by market |

---

## Future Enhancements

- [ ] **GARCH(1,1)** volatility forecasting for time-varying σ estimates
- [ ] **Student-t / skew-t** simulation for heavier tails
- [ ] **Copula-based** dependence (Clayton, Gumbel) to model tail dependence separately from correlation
- [ ] **Delta-normal VaR** for options/derivatives overlays on the equity positions
- [ ] **VaR backtest** with Basel traffic-light exception counting (green/yellow/red zones)
- [ ] **PCA factor decomposition** of the covariance matrix
- [ ] **Incremental VaR** — impact of adding/removing a single position
- [ ] **Multi-currency** portfolio with FX risk layer

---

## Tech Stack

| Library | Role |
|---|---|
| `numpy` | Simulation core, Cholesky decomposition, matrix operations |
| `pandas` | Price data, return frames, structured CSV output |
| `matplotlib` | All 9 charts — no seaborn dependency |
| `scipy` | Parametric VaR/CVaR, distribution statistics |
| `yfinance` | Historical adjusted-close price download |
| `streamlit` | Interactive browser dashboard |
| `pytest` | 50-test unit test suite |
| `tabulate` | Markdown table formatting in risk reports |

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built as a quantitative portfolio project targeting risk, derivatives, and financial engineering roles.*
