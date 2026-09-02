"""
visualization.py
================
Publication-quality chart generation for the Monte Carlo Portfolio Risk
Simulator.  Every figure is self-contained, saves to disk, and is designed
to communicate clearly to both quant practitioners and non-technical
stakeholders.

Chart catalogue
---------------
1.  plot_return_distribution   — simulated P&L histogram with VaR/CVaR overlays
2.  plot_pnl_distribution      — dollar P&L with VaR markers
3.  plot_correlation_heatmap   — Pearson correlation heatmap
4.  plot_cumulative_returns    — historical equity curves
5.  plot_simulation_paths      — Monte Carlo fan with percentile envelope
6.  plot_stress_comparison     — grouped bar: baseline vs stressed VaR/CVaR
7.  plot_multi_horizon_var     — VaR scaling across 1-day / 10-day / 252-day
8.  plot_component_var         — per-asset VaR contribution (risk attribution)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

from utils import get_logger, ensure_dirs

matplotlib.use("Agg")   # non-interactive backend — safe for headless / CI runs

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared design system
# ---------------------------------------------------------------------------

PALETTE = {
    "primary":   "#1f4e79",   # deep navy  — portfolio / main series
    "secondary": "#2e86c1",   # mid blue   — simulated distribution
    "accent":    "#c0392b",   # crimson    — 99 % risk markers
    "accent2":   "#d35400",   # burnt orange — 95 % risk markers
    "positive":  "#1e8449",   # forest green — upside / gains
    "neutral":   "#6c757d",   # medium grey — reference lines
    "muted":     "#adb5bd",   # light grey  — secondary grid
    "bg":        "#f7f9fc",   # near-white  — figure background
    "panel":     "#eef2f7",   # light panel — inset / annotation boxes
}

_FOOTER = "Monte Carlo Portfolio Risk Simulator  ·  Multivariate Normal / Cholesky Decomposition"


def _apply_style(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    subtitle: str = "",
) -> None:
    """Apply a consistent, professional style to an Axes object."""
    ax.set_xlabel(xlabel, fontsize=11, labelpad=8, color="#2c3e50")
    ax.set_ylabel(ylabel, fontsize=11, labelpad=8, color="#2c3e50")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["muted"])
    ax.spines["bottom"].set_color(PALETTE["muted"])
    ax.grid(axis="y", linestyle="--", alpha=0.45, linewidth=0.75, color=PALETTE["muted"])
    ax.tick_params(labelsize=10, color=PALETTE["neutral"])
    ax.set_facecolor(PALETTE["bg"])

    if subtitle:
        ax.set_title(
            f"{title}\n",
            fontsize=14, fontweight="bold", pad=4, color="#1a1a2e", loc="left",
        )
        ax.annotate(
            subtitle, xy=(0, 1.01), xycoords="axes fraction",
            fontsize=9, color=PALETTE["neutral"], va="bottom",
        )
    else:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10,
                     color="#1a1a2e", loc="left")


def _add_footer(fig: plt.Figure) -> None:
    """Add a small branding footer to every figure."""
    fig.text(
        0.99, 0.005, _FOOTER,
        ha="right", va="bottom", fontsize=7.5,
        color=PALETTE["neutral"], style="italic",
    )


def _save(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    ensure_dirs(str(Path(path).parent))
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=PALETTE["bg"], edgecolor="none")
    plt.close(fig)
    logger.info("Saved chart → %s", path)


# ---------------------------------------------------------------------------
# 1. Return distribution histogram
# ---------------------------------------------------------------------------

def plot_return_distribution(
    returns: np.ndarray,
    var_95: float,
    var_99: float,
    cvar_95: float,
    cvar_99: float,
    horizon_label: str,
    out_path: str,
) -> None:
    """
    Histogram of simulated portfolio log-returns with VaR and CVaR overlays.

    The left tail region beyond 99 % VaR is shaded to highlight the Expected
    Shortfall (CVaR) zone — the region that Basel III's FRTB framework
    requires banks to capitalise against.
    """
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=PALETTE["bg"])

    n_bins = min(120, max(60, len(returns) // 100))
    counts, edges, patches = ax.hist(
        returns, bins=n_bins,
        color=PALETTE["secondary"], alpha=0.68,
        edgecolor="white", linewidth=0.25,
        density=True, label="Simulated Returns",
    )

    # Colour the left tail bars differently for visual emphasis
    threshold = -var_99
    for patch, left in zip(patches, edges[:-1]):
        if left < threshold:
            patch.set_facecolor(PALETTE["accent"])
            patch.set_alpha(0.55)

    # VaR / CVaR vertical lines
    line_kw = dict(lw=1.9, solid_capstyle="round")
    ax.axvline(-var_95,  color=PALETTE["accent2"], ls="--", **line_kw,
               label=f"95% VaR  = {var_95:.2%}")
    ax.axvline(-var_99,  color=PALETTE["accent"],  ls="--", **line_kw,
               label=f"99% VaR  = {var_99:.2%}")
    ax.axvline(-cvar_95, color=PALETTE["accent2"], ls=":",  lw=1.7,
               label=f"95% CVaR = {cvar_95:.2%}")
    ax.axvline(-cvar_99, color=PALETTE["accent"],  ls=":",  lw=1.7,
               label=f"99% CVaR = {cvar_99:.2%}")

    # Annotation: CVaR region label
    y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else counts.max() * 1.05
    ax.annotate(
        "Expected\nShortfall\nZone",
        xy=(-var_99 - (returns.max() - returns.min()) * 0.01, y_top * 0.55),
        fontsize=8, color=PALETTE["accent"], ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc=PALETTE["panel"], ec=PALETTE["accent"],
                  alpha=0.8, lw=0.7),
    )

    _apply_style(
        ax,
        title=f"Simulated Portfolio Return Distribution  —  {horizon_label} Horizon",
        subtitle=f"{len(returns):,} Monte Carlo paths  ·  Multivariate Normal (Cholesky)  ·  "
                 f"Left tail = loss scenarios",
        xlabel="Portfolio Log-Return",
        ylabel="Probability Density",
    )
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.legend(fontsize=9, framealpha=0.9, edgecolor=PALETTE["muted"],
              loc="upper left")
    _add_footer(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 2. P&L histogram (dollar losses)
# ---------------------------------------------------------------------------

def plot_pnl_distribution(
    returns: np.ndarray,
    portfolio_value: float,
    var_95: float,
    var_99: float,
    out_path: str,
) -> None:
    """
    Histogram of simulated dollar P&L with VaR markers and loss/gain split.

    Directly maps to the daily P&L report format used by risk desks.
    """
    pnl = returns * portfolio_value
    var_95_usd = var_95 * portfolio_value
    var_99_usd = var_99 * portfolio_value

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=PALETTE["bg"])
    n_bins = min(120, max(60, len(pnl) // 100))

    # Split into gain/loss colours for instant readability
    ax.hist(pnl[pnl >= 0], bins=n_bins // 2, color=PALETTE["positive"],
            alpha=0.65, density=False, edgecolor="white", linewidth=0.25,
            label="Gain scenarios")
    ax.hist(pnl[pnl <  0], bins=n_bins // 2, color=PALETTE["primary"],
            alpha=0.65, density=False, edgecolor="white", linewidth=0.25,
            label="Loss scenarios")

    ax.axvline(-var_95_usd, color=PALETTE["accent2"], lw=2.0, ls="--",
               label=f"95% VaR = ${var_95_usd:,.0f}")
    ax.axvline(-var_99_usd, color=PALETTE["accent"],  lw=2.0, ls="--",
               label=f"99% VaR = ${var_99_usd:,.0f}")
    ax.axvline(0, color=PALETTE["neutral"], lw=1.2, ls="-", alpha=0.6)

    pct_loss = (pnl < 0).mean()
    ax.annotate(
        f"{pct_loss:.1%} loss days",
        xy=(-var_95_usd * 0.5, ax.get_ylim()[1] * 0.8 if ax.get_ylim()[1] > 0 else 1),
        fontsize=9, color=PALETTE["primary"], ha="center",
    )

    _apply_style(
        ax,
        title="Simulated Portfolio P&L Distribution  —  1-Day Horizon",
        subtitle=f"Notional: USD {portfolio_value:,.0f}  ·  "
                 f"Loss probability: {pct_loss:.1%}  ·  "
                 f"Mean daily P&L: USD {pnl.mean():,.0f}",
        xlabel="Daily P&L  (USD)",
        ylabel="Scenario Count",
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9, framealpha=0.9, edgecolor=PALETTE["muted"])
    _add_footer(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 3. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    out_path: str,
) -> None:
    """
    Pearson correlation heatmap with diverging colour scale.

    High equity-equity correlation concentrates risk; negative equity-bond
    correlation is the foundation of the classic 60/40 diversification argument.
    """
    n = len(corr_matrix)
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=PALETTE["bg"])

    # Diverging palette: red = positive, white = zero, green = negative
    cmap = plt.cm.RdYlGn_r   # reverse so high correlation = warm/red
    im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    for i in range(n):
        for j in range(n):
            val = corr_matrix.values[i, j]
            text_color = "white" if abs(val) > 0.55 else "#2c3e50"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10.5, fontweight=weight, color=text_color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr_matrix.columns, fontsize=11, rotation=35, ha="right")
    ax.set_yticklabels(corr_matrix.index,   fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation  (daily log-returns)", fontsize=9)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(
        "Asset Return Correlation Matrix",
        fontsize=14, fontweight="bold", pad=12, color="#1a1a2e", loc="left",
    )
    ax.annotate(
        "High off-diagonal values indicate concentrated risk; "
        "low / negative values indicate diversification benefit.",
        xy=(0, -0.14), xycoords="axes fraction",
        fontsize=8.5, color=PALETTE["neutral"], va="top",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    _add_footer(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 4. Cumulative return (historical)
# ---------------------------------------------------------------------------

def plot_cumulative_returns(
    prices: pd.DataFrame,
    weights: np.ndarray,
    out_path: str,
) -> None:
    """
    Cumulative growth of each constituent asset and the blended portfolio.

    Provides historical context for the risk estimates — a portfolio that
    survived 2020 Covid drawdown and 2022 rate-shock is a credible benchmark.
    """
    log_rets   = np.log(prices / prices.shift(1)).dropna()
    cum_assets = np.exp(log_rets.cumsum())
    port_rets  = (log_rets * weights).sum(axis=1)
    cum_port   = np.exp(port_rets.cumsum())

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=PALETTE["bg"])

    colors = plt.cm.tab10(np.linspace(0, 0.85, len(prices.columns)))
    for col, c in zip(cum_assets.columns, colors):
        ax.plot(cum_assets.index, cum_assets[col],
                lw=1.3, alpha=0.6, color=c, label=col)

    ax.plot(cum_port.index, cum_port.values,
            lw=2.8, color=PALETTE["primary"], label="Blended Portfolio", zorder=5)
    ax.axhline(1.0, color=PALETTE["neutral"], lw=1.0, ls="--", alpha=0.5)

    # Annotate final values
    for col, c in zip(cum_assets.columns, colors):
        final = float(cum_assets[col].iloc[-1])
        ax.annotate(f"{final:.2f}×", xy=(cum_assets.index[-1], final),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=8, color=c, va="center")
    final_port = float(cum_port.iloc[-1])
    ax.annotate(f"{final_port:.2f}×", xy=(cum_port.index[-1], final_port),
                xytext=(5, 0), textcoords="offset points",
                fontsize=9, fontweight="bold", color=PALETTE["primary"], va="center")

    start_yr = prices.index[0].year
    end_yr   = prices.index[-1].year
    _apply_style(
        ax,
        title=f"Historical Cumulative Returns  ({start_yr}–{end_yr})",
        subtitle="Adjusted-close prices  ·  Log-return compounding  ·  "
                 "Base = $1.00 at inception",
        xlabel="Date",
        ylabel="Growth of $1 Invested",
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}×"))
    ax.legend(fontsize=9, framealpha=0.9, edgecolor=PALETTE["muted"], ncol=2)
    _add_footer(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 5. Monte Carlo simulation paths
# ---------------------------------------------------------------------------

def plot_simulation_paths(
    cumulative_paths: np.ndarray,
    horizon: int,
    n_plot: int,
    portfolio_value: float,
    var_95: float,
    out_path: str,
) -> None:
    """
    Fan chart of Monte Carlo cumulative wealth paths with percentile envelope.

    The dispersion of paths widens with horizon — consistent with the
    √T scaling of volatility — and the envelope quantifies the range of
    plausible outcomes used for long-horizon stress analysis.
    """
    if cumulative_paths is None:
        logger.warning("No cumulative paths available for plotting.")
        return

    n_available = cumulative_paths.shape[1]
    steps = np.arange(cumulative_paths.shape[0])

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=PALETTE["bg"])

    # Sample paths (background)
    idx = np.random.default_rng(99).choice(
        n_available, size=min(n_plot, n_available), replace=False
    )
    for i in idx:
        ax.plot(steps, cumulative_paths[:, i] * portfolio_value,
                lw=0.45, alpha=0.18, color=PALETTE["secondary"])

    # Percentile envelopes
    p1  = np.percentile(cumulative_paths, 1,  axis=1) * portfolio_value
    p5  = np.percentile(cumulative_paths, 5,  axis=1) * portfolio_value
    p25 = np.percentile(cumulative_paths, 25, axis=1) * portfolio_value
    p50 = np.percentile(cumulative_paths, 50, axis=1) * portfolio_value
    p75 = np.percentile(cumulative_paths, 75, axis=1) * portfolio_value
    p95 = np.percentile(cumulative_paths, 95, axis=1) * portfolio_value

    ax.fill_between(steps, p5, p95, alpha=0.12, color=PALETTE["primary"],
                    label="5th–95th Percentile band")
    ax.fill_between(steps, p25, p75, alpha=0.18, color=PALETTE["primary"],
                    label="25th–75th Percentile band")
    ax.plot(steps, p50, lw=2.5, color=PALETTE["primary"],  label="Median (50th Pct)", zorder=5)
    ax.plot(steps, p5,  lw=1.6, color=PALETTE["accent"],   label="5th Percentile",    zorder=5)
    ax.plot(steps, p1,  lw=1.2, color=PALETTE["accent"],   ls=":", alpha=0.8,
            label="1st Percentile", zorder=5)
    ax.plot(steps, p95, lw=1.6, color=PALETTE["positive"], label="95th Percentile",   zorder=5)
    ax.axhline(portfolio_value, color=PALETTE["neutral"], lw=1.5, ls="--",
               label="Initial Value", alpha=0.65)

    # Annotate terminal percentile values
    for val, label, color in [
        (p95[-1], "95th", PALETTE["positive"]),
        (p50[-1], "Median", PALETTE["primary"]),
        (p5[-1],  "5th",  PALETTE["accent"]),
    ]:
        ax.annotate(f"${val:,.0f}", xy=(steps[-1], val),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=8.5, color=color, va="center")

    _apply_style(
        ax,
        title=f"Monte Carlo Portfolio Paths  —  {horizon}-Day Horizon",
        subtitle=f"{n_available:,} simulated paths  ·  Shaded bands = interquartile & 5th–95th ranges  ·  "
                 f"Notional: ${portfolio_value:,.0f}",
        xlabel="Trading Days",
        ylabel="Portfolio Value  (USD)",
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9, framealpha=0.9, edgecolor=PALETTE["muted"], ncol=2,
              loc="upper left")
    _add_footer(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 6. Stress scenario comparison
# ---------------------------------------------------------------------------

def plot_stress_comparison(
    scenario_df: pd.DataFrame,
    confidence: str,
    out_path: str,
) -> None:
    """
    Grouped bar chart comparing VaR and CVaR across all scenarios.

    The delta from baseline visually captures how much additional capital
    would be required to cover tail risk under each stress regime — the
    core question of a stress-testing framework (DFAST, CCAR, EBA).
    """
    sub = scenario_df[scenario_df["Confidence"] == confidence].copy()

    def _p(s: str) -> float:
        return float(str(s).replace("%", "").strip()) / 100

    sub["var_val"]  = sub["VaR (%)"].apply(_p)
    sub["cvar_val"] = sub["CVaR (%)"].apply(_p)

    scenarios = sub["Scenario"].tolist()
    x = np.arange(len(scenarios))
    width = 0.34

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=PALETTE["bg"])

    bars1 = ax.bar(x - width / 2, sub["var_val"],  width,
                   label=f"VaR  ({confidence})",
                   color=PALETTE["secondary"], alpha=0.87, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, sub["cvar_val"], width,
                   label=f"CVaR ({confidence})  [Expected Shortfall]",
                   color=PALETTE["accent"],   alpha=0.87, edgecolor="white", linewidth=0.5)

    # Highlight baseline differently
    if scenarios and scenarios[0] == "Baseline":
        bars1[0].set_color(PALETTE["neutral"])
        bars2[0].set_color(PALETTE["neutral"])
        bars1[0].set_alpha(0.6)
        bars2[0].set_alpha(0.6)

    # Value labels
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.00015,
                f"{h:.2%}", ha="center", va="bottom", fontsize=8,
                color="#2c3e50", fontweight="semibold")

    # Reference line at baseline CVaR
    if len(sub) > 0:
        base_cvar = float(sub.iloc[0]["cvar_val"])
        ax.axhline(base_cvar, color=PALETTE["neutral"], lw=1.2, ls="--", alpha=0.7,
                   label=f"Baseline CVaR = {base_cvar:.2%}")

    _apply_style(
        ax,
        title=f"Stress Scenario Risk Comparison  —  {confidence} Confidence",
        subtitle="Reflects DFAST / CCAR-style instantaneous shock methodology  ·  "
                 "CVaR = Expected Shortfall (Basel III FRTB preferred metric)",
        xlabel="Scenario",
        ylabel="Portfolio Loss  (% of Notional)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=18, ha="right", fontsize=9.5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=2))
    ax.legend(fontsize=9.5, framealpha=0.9, edgecolor=PALETTE["muted"])
    _add_footer(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 7. Multi-horizon VaR comparison
# ---------------------------------------------------------------------------

def plot_multi_horizon_var(
    horizon_results: dict,
    confidence_levels: list[float],
    portfolio_value: float,
    out_path: str,
) -> None:
    """
    VaR in USD across 1-day, 10-day, and 252-day horizons at each confidence.

    Demonstrates the √T scaling property: 10-day VaR ≈ 1-day VaR × √10.
    The Basel II/III market risk capital charge uses a 10-day 99 % VaR.
    """
    from risk_metrics import compute_var

    line_colors = [PALETTE["secondary"], PALETTE["accent"]]
    markers = ["o", "s"]
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=PALETTE["bg"])

    labels_order = list(horizon_results.keys())
    for cl, col, mk in zip(confidence_levels, line_colors, markers):
        var_vals = [
            compute_var(horizon_results[lbl].portfolio_returns, cl) * portfolio_value
            for lbl in labels_order
        ]
        ax.plot(labels_order, var_vals, marker=mk, lw=2.2, color=col,
                markersize=9, markerfacecolor="white", markeredgewidth=2,
                label=f"{cl:.0%} VaR")
        for lbl, val in zip(labels_order, var_vals):
            ax.annotate(
                f"${val:,.0f}",
                xy=(lbl, val), xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=9, color=col, fontweight="semibold",
            )

    # Overlay a theoretical √T line anchored at 1-day
    first_1d = compute_var(
        horizon_results[labels_order[0]].portfolio_returns, confidence_levels[0]
    ) * portfolio_value
    horizons_days = [h for h in [1, 10, 252] if h <= 252]
    sqrt_t_vals   = [first_1d * np.sqrt(d) for d in horizons_days]
    ax.plot(labels_order[:len(sqrt_t_vals)], sqrt_t_vals,
            lw=1.3, ls=":", color=PALETTE["neutral"], alpha=0.7,
            label="√T scaling (theoretical)")

    _apply_style(
        ax,
        title="VaR Scaling Across Time Horizons",
        subtitle="Basel II/III 10-day 99 % VaR is the standard market-risk capital metric  ·  "
                 "Dotted line = theoretical √T scaling",
        xlabel="Time Horizon",
        ylabel="Value at Risk  (USD)",
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=10, framealpha=0.9, edgecolor=PALETTE["muted"])
    _add_footer(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 8. Component VaR attribution (risk decomposition)
# ---------------------------------------------------------------------------

def plot_component_var(
    component_var_series: pd.Series,
    total_var_usd: float,
    confidence: float,
    out_path: str,
) -> None:
    """
    Horizontal bar chart of per-asset VaR contribution.

    Component VaR decomposes total portfolio VaR into each asset's marginal
    contribution, accounting for cross-asset correlations.  An asset with a
    small weight but high correlation to the rest of the portfolio may
    contribute disproportionately to risk — critical insight for hedging
    and position-sizing decisions.

    Parameters
    ----------
    component_var_series : pd.Series indexed by ticker
    total_var_usd        : total portfolio VaR in USD (for percentage labelling)
    confidence           : e.g. 0.95
    """
    tickers = component_var_series.index.tolist()
    values  = component_var_series.values
    pct_of_total = values / total_var_usd * 100

    # Sort descending
    order = np.argsort(values)[::-1]
    tickers_s = [tickers[i] for i in order]
    values_s  = values[order]
    pct_s     = pct_of_total[order]

    # Color gradient: largest contributor = darkest primary
    n = len(tickers_s)
    bar_colors = [PALETTE["primary"] if i == 0 else PALETTE["secondary"]
                  if pct_s[i] >= 15 else PALETTE["muted"]
                  for i in range(n)]

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.9 + 1.5)), facecolor=PALETTE["bg"])
    bars = ax.barh(tickers_s, values_s, color=bar_colors,
                   alpha=0.87, edgecolor="white", linewidth=0.5, height=0.55)

    for bar, val, pct in zip(bars, values_s, pct_s):
        ax.text(val + total_var_usd * 0.008, bar.get_y() + bar.get_height() / 2,
                f"${val:,.0f}  ({pct:.1f}%)",
                va="center", fontsize=9.5, color="#2c3e50")

    ax.axvline(0, color=PALETTE["neutral"], lw=0.8)
    ax.invert_yaxis()

    _apply_style(
        ax,
        title=f"Component VaR Attribution  —  {confidence:.0%} Confidence",
        subtitle="Shows each asset's marginal contribution to total portfolio VaR  ·  "
                 "Components sum to total VaR  ·  Key input for hedging overlay design",
        xlabel="VaR Contribution  (USD)",
        ylabel="",
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(axis="x", linestyle="--", alpha=0.4, linewidth=0.75, color=PALETTE["muted"])
    ax.grid(axis="y", visible=False)

    # Summary annotation box
    ax.annotate(
        f"Total Portfolio {confidence:.0%} VaR\n${total_var_usd:,.0f}",
        xy=(0.98, 0.05), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=9.5, fontweight="bold",
        color=PALETTE["primary"],
        bbox=dict(boxstyle="round,pad=0.4", fc=PALETTE["panel"],
                  ec=PALETTE["primary"], lw=1.0, alpha=0.9),
    )
    _add_footer(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    _save(fig, out_path)
