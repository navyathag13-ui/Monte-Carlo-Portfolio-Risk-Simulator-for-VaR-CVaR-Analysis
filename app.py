"""
app.py
======
Streamlit dashboard for the Monte Carlo Portfolio Risk Simulator.

Launch with:
    streamlit run app.py
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st

matplotlib.use("Agg")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Monte Carlo Portfolio Risk Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Portfolio Inputs
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("📊 Portfolio Configuration")
st.sidebar.markdown("---")

tickers_raw = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value="SPY, QQQ, TLT, GLD, AAPL",
)
tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

weights_raw = st.sidebar.text_input(
    "Weights (comma-separated, must sum to 1)",
    value="0.30, 0.25, 0.20, 0.15, 0.10",
)
try:
    weights = [float(w.strip()) for w in weights_raw.split(",")]
except ValueError:
    weights = []

portfolio_value = st.sidebar.number_input(
    "Portfolio Value ($)",
    min_value=10_000,
    max_value=1_000_000_000,
    value=1_000_000,
    step=10_000,
    format="%d",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Settings")
n_sims = st.sidebar.select_slider(
    "Number of Simulations",
    options=[1_000, 5_000, 10_000, 20_000, 50_000],
    value=10_000,
)
random_seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, max_value=9999)

st.sidebar.markdown("---")
st.sidebar.subheader("Historical Data Range")
start_date = st.sidebar.text_input("Start Date", value="2018-01-01")
end_date   = st.sidebar.text_input("End Date",   value="2023-12-31")

confidence_95 = st.sidebar.checkbox("Show 95% VaR/CVaR", value=True)
confidence_99 = st.sidebar.checkbox("Show 99% VaR/CVaR", value=True)
confidence_levels = [c for c, flag in [(0.95, confidence_95), (0.99, confidence_99)] if flag]
if not confidence_levels:
    confidence_levels = [0.95]

run_button = st.sidebar.button("▶  Run Simulation", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main title
# ─────────────────────────────────────────────────────────────────────────────

st.title("Monte Carlo Portfolio Risk Simulator")
st.markdown(
    "**VaR · CVaR · Stress Testing · Scenario Analysis** — "
    "a quantitative risk tool for multi-asset portfolios."
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_inputs():
    errors = []
    if len(tickers) == 0:
        errors.append("Please enter at least one ticker.")
    if len(weights) != len(tickers):
        errors.append(f"Mismatch: {len(tickers)} tickers but {len(weights)} weights.")
    elif any(w < 0 for w in weights):
        errors.append("All weights must be ≥ 0.")
    elif not np.isclose(sum(weights), 1.0, atol=0.01):
        errors.append(f"Weights sum to {sum(weights):.3f} — they must sum to 1.0.")
    return errors


errors = _validate_inputs()
if errors:
    for e in errors:
        st.error(e)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Simulation (cached for performance)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _run_simulation(
    _tickers, _weights, _value, _n_sims, _seed, _start, _end, _confidence_levels
):
    """Cached simulation runner — parameters form the cache key."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from data_loader import load_price_data
    from portfolio import build_portfolio
    from simulation import run_all_horizons
    from risk_metrics import build_risk_table, risk_at_confidence
    from stress_testing import run_all_stress_scenarios, worst_scenario
    import config

    prices    = load_price_data(_tickers, _start, _end, random_seed=_seed)
    portfolio = build_portfolio(_tickers, list(_weights), prices, portfolio_value=_value)

    horizons = {"1-day": 1, "10-day": 10, "252-day": 252}
    horizon_results = run_all_horizons(
        portfolio=portfolio,
        horizons=horizons,
        n_simulations=_n_sims,
        random_seed=_seed,
        full_paths_horizon="252-day",
    )

    mc_1d = horizon_results["1-day"].portfolio_returns
    hist  = portfolio.historical_portfolio_returns().values

    risk_df = build_risk_table(mc_1d, hist, _confidence_levels, _value)
    r_dict  = {cl: risk_at_confidence(mc_1d, cl, _value) for cl in _confidence_levels}

    scenario_df = run_all_stress_scenarios(
        portfolio=portfolio,
        scenarios=config.STRESS_SCENARIOS,
        confidence_levels=_confidence_levels,
        horizon=1,
        n_simulations=_n_sims,
        random_seed=_seed,
        portfolio_value=_value,
    )
    worst = worst_scenario(scenario_df, confidence=f"{_confidence_levels[0]:.0%}")

    return portfolio, horizon_results, risk_df, r_dict, scenario_df, worst


# ─────────────────────────────────────────────────────────────────────────────
# Display results on button click (or initial state)
# ─────────────────────────────────────────────────────────────────────────────

if run_button or True:   # always show on load with defaults
    with st.spinner("Running Monte Carlo simulation …"):
        try:
            (
                portfolio, horizon_results, risk_df, r_dict,
                scenario_df, worst_name,
            ) = _run_simulation(
                tuple(tickers), tuple(weights), portfolio_value,
                n_sims, random_seed, start_date, end_date,
                tuple(confidence_levels),
            )
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
            st.stop()

    # ── KPI cards ────────────────────────────────────────────────────────────
    st.subheader("Portfolio Statistics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ann. Return",    f"{portfolio.annualised_return:.2%}")
    c2.metric("Ann. Volatility",f"{portfolio.annualised_volatility:.2%}")
    c3.metric("Sharpe Ratio",   f"{portfolio.sharpe_ratio:.3f}")
    c4.metric("Observations",   f"{len(portfolio.returns):,}")
    c5.metric("Simulations",    f"{n_sims:,}")

    st.markdown("---")

    # ── Risk metrics table ───────────────────────────────────────────────────
    st.subheader("Risk Metrics — One-Day Horizon")
    st.dataframe(risk_df, use_container_width=True, hide_index=True)

    # ── Large KPI row ────────────────────────────────────────────────────────
    if 0.95 in r_dict:
        r95 = r_dict[0.95]
        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric("95% VaR",    f"{r95['var_pct']:.3%}",  f"-${r95['var_usd']:,.0f}")
        kc2.metric("95% CVaR",   f"{r95['cvar_pct']:.3%}", f"-${r95['cvar_usd']:,.0f}")
    if 0.99 in r_dict:
        r99 = r_dict[0.99]
        kc3.metric("99% VaR",    f"{r99['var_pct']:.3%}",  f"-${r99['var_usd']:,.0f}")
        kc4.metric("99% CVaR",   f"{r99['cvar_pct']:.3%}", f"-${r99['cvar_usd']:,.0f}")

    st.markdown("---")

    # ── Charts ───────────────────────────────────────────────────────────────
    st.subheader("Visualisations")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Return Distribution", "Simulation Paths",
        "Correlation Matrix", "Cumulative Returns", "Stress Scenarios",
    ])

    # helper to render matplotlib fig in streamlit
    def _render(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        buf.seek(0)
        st.image(buf)
        plt.close(fig)

    with tab1:
        from visualization import plot_return_distribution
        mc_1d = horizon_results["1-day"].portfolio_returns
        r95 = r_dict.get(0.95, {"var_pct": 0, "cvar_pct": 0, "var_usd": 0, "cvar_usd": 0})
        r99 = r_dict.get(0.99, r95)
        fig, ax = plt.subplots(figsize=(11, 5), facecolor="#f8f9fa")
        n_bins = min(100, len(mc_1d) // 80)
        ax.hist(mc_1d, bins=n_bins, color="#2e86c1", alpha=0.7, density=True,
                edgecolor="white", linewidth=0.3)
        ax.axvline(-r95["var_pct"], color="#f39c12", lw=2, ls="--",
                   label=f"95% VaR = {r95['var_pct']:.2%}")
        ax.axvline(-r99["var_pct"], color="#e74c3c", lw=2, ls="--",
                   label=f"99% VaR = {r99['var_pct']:.2%}")
        ax.set_title("Simulated 1-Day Portfolio Return Distribution", fontsize=13, fontweight="bold")
        ax.set_xlabel("Portfolio Log-Return")
        ax.set_ylabel("Density")
        import matplotlib.ticker as mticker
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend()
        _render(fig)

    with tab2:
        result_252 = horizon_results["252-day"]
        if result_252.cumulative_paths is not None:
            cp = result_252.cumulative_paths * portfolio_value
            steps = np.arange(cp.shape[0])
            fig, ax = plt.subplots(figsize=(11, 5), facecolor="#f8f9fa")
            idx = np.random.default_rng(77).choice(cp.shape[1], size=min(150, cp.shape[1]),
                                                   replace=False)
            for i in idx:
                ax.plot(steps, cp[:, i], lw=0.4, alpha=0.2, color="#2e86c1")
            p5  = np.percentile(cp, 5,  axis=1)
            p50 = np.percentile(cp, 50, axis=1)
            p95 = np.percentile(cp, 95, axis=1)
            ax.fill_between(steps, p5, p95, alpha=0.15, color="#1f4e79")
            ax.plot(steps, p50, lw=2.5, color="#1f4e79", label="Median")
            ax.plot(steps, p5,  lw=1.5, color="#e74c3c", label="5th Percentile")
            ax.plot(steps, p95, lw=1.5, color="#27ae60", label="95th Percentile")
            ax.axhline(portfolio_value, color="#7f8c8d", lw=1.5, ls="--", label="Initial Value")
            ax.set_title("Monte Carlo Paths — 252-Day Horizon", fontsize=13, fontweight="bold")
            ax.set_xlabel("Trading Days")
            ax.set_ylabel("Portfolio Value ($)")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend()
            _render(fig)

    with tab3:
        corr = portfolio.corr_matrix
        n = len(corr)
        fig, ax = plt.subplots(figsize=(7, 6), facecolor="#f8f9fa")
        im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
        for i in range(n):
            for j in range(n):
                v = corr.values[i, j]
                color = "white" if abs(v) > 0.6 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr.columns, rotation=30)
        ax.set_yticklabels(corr.index)
        plt.colorbar(im, ax=ax)
        ax.set_title("Asset Correlation Matrix", fontsize=13, fontweight="bold")
        _render(fig)

    with tab4:
        log_r = np.log(portfolio.prices / portfolio.prices.shift(1)).dropna()
        cum   = np.exp(log_r.cumsum())
        port_r = (log_r * portfolio.weights).sum(axis=1)
        cum_port = np.exp(port_r.cumsum())
        fig, ax = plt.subplots(figsize=(11, 5), facecolor="#f8f9fa")
        colors = plt.cm.tab10(np.linspace(0, 1, len(portfolio.tickers)))
        for col, c in zip(cum.columns, colors):
            ax.plot(cum.index, cum[col], lw=1.2, alpha=0.6, color=c, label=col)
        ax.plot(cum_port.index, cum_port.values, lw=2.5, color="#1f4e79",
                label="Portfolio", zorder=5)
        ax.set_title("Cumulative Returns (Historical)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Growth of $1")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}×"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(ncol=3)
        _render(fig)

    with tab5:
        cl_label = f"{confidence_levels[0]:.0%}"
        sub = scenario_df[scenario_df["Confidence"] == cl_label].copy()

        def _p(s):
            return float(str(s).replace("%", "").strip()) / 100

        sub["var_val"]  = sub["VaR (%)"].apply(_p)
        sub["cvar_val"] = sub["CVaR (%)"].apply(_p)

        x = np.arange(len(sub))
        fig, ax = plt.subplots(figsize=(11, 5), facecolor="#f8f9fa")
        ax.bar(x - 0.2, sub["var_val"],  0.35, label=f"VaR ({cl_label})",  color="#2e86c1", alpha=0.85)
        ax.bar(x + 0.2, sub["cvar_val"], 0.35, label=f"CVaR ({cl_label})", color="#e74c3c", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["Scenario"].tolist(), rotation=20, ha="right")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=2))
        ax.set_title(f"Stress Scenario Comparison — {cl_label}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Portfolio Loss (%)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend()
        _render(fig)
        st.info(f"**Worst stress scenario ({cl_label} CVaR):** {worst_name}")

    st.markdown("---")

    # ── Stress detail table ──────────────────────────────────────────────────
    st.subheader("Stress Scenario Detail")
    st.dataframe(scenario_df, use_container_width=True, hide_index=True)

    # ── Downloadable CSV ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Download Results")
    dc1, dc2 = st.columns(2)
    dc1.download_button(
        "⬇ Risk Metrics CSV",
        data=risk_df.to_csv(index=False),
        file_name="risk_metrics.csv",
        mime="text/csv",
    )
    dc2.download_button(
        "⬇ Stress Scenarios CSV",
        data=scenario_df.to_csv(index=False),
        file_name="stress_scenarios.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.caption(
        "Monte Carlo Portfolio Risk Simulator · "
        "Built with Python, NumPy, pandas, Matplotlib, and Streamlit · "
        "For educational and analytical purposes only."
    )
