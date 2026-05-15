"""
analysis/reporter.py

Generates a self-contained HTML report summarising all pipeline results.

The report contains:
    1. Executive summary — top leading CAMEO categories per commodity
    2. Granger heatmap — p-value (FDR-adjusted) grid: roots × commodities × lags
    3. Correlation heatmap — Pearson r at lag-5 days (primary lead horizon)
    4. GSI time series — with commodity volatility overlay
    5. Rolling 90-day GSI-volatility correlation chart
    6. Stationarity audit table
    7. Methodology notes

All charts are generated with Plotly (interactive) and embedded as base64
in a single standalone HTML file (no external dependencies at view time).
"""

import base64
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
from nlp.cameo_codes import CAMEO_ROOTS, ALL_ROOTS
from nlp.stress_index import load_gsi

log = logging.getLogger(__name__)

_ROLLING_CORR_PATH = config.PROCESSED_DIR / "gsi_rolling_corr.parquet"
_STATIONARITY_PATH = config.RESULTS_DIR / "stationarity_report.csv"


# ── Chart builders ────────────────────────────────────────────────────────────

def _granger_heatmap_html(granger_df: pd.DataFrame, direction: str = "cameo→vol") -> str:
    """Return a Plotly heatmap HTML div showing −log10(q_value) per root × commodity."""
    try:
        import plotly.graph_objects as go
        import plotly.offline as pyo
    except ImportError:
        return "<p><em>Plotly not available — install plotly to see charts.</em></p>"

    df = granger_df[granger_df["direction"] == direction].copy()
    df = df[df["lag"] == 5]  # Show lag=5 as the primary "1-week-ahead" view

    # Pivot: rows = CAMEO roots, cols = commodities
    pivot = df.pivot_table(
        index="cameo_root", columns="commodity",
        values="q_value_fdr", aggfunc="min"
    )

    # Use −log10(q) so high values = more significant
    z = -np.log10(pivot.clip(lower=1e-10).values)
    labels = [[f"q={pivot.iloc[i,j]:.3f}" for j in range(pivot.shape[1])]
              for i in range(pivot.shape[0])]

    root_labels = [
        f"{r}: {CAMEO_ROOTS.get(str(r), type('', (), {'label': str(r)})()).label[:28]}"
        for r in pivot.index
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[config.TICKER_NAMES.get(c, c) for c in pivot.columns],
        y=root_labels,
        text=labels,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmin=0,
        zmax=3,
        colorbar=dict(title="−log₁₀(q)", tickvals=[0, 1, 2, 3],
                      ticktext=["q=1.0", "q=0.1", "q=0.01", "q=0.001"]),
    ))
    fig.update_layout(
        title=f"Granger Causality Heatmap — {direction} (lag=5 days, FDR-adjusted q)",
        xaxis_title="Commodity",
        yaxis_title="CAMEO Root Category",
        height=700,
        margin=dict(l=320),
    )
    # Add significance boundary line at q=0.05 → −log10(0.05)=1.301
    fig.add_annotation(
        text="── q=0.05 boundary ──",
        xref="paper", yref="paper", x=1.02, y=0.43,
        showarrow=False, font=dict(size=10, color="red"),
    )

    return pyo.plot(fig, output_type="div", include_plotlyjs=False)


def _correlation_heatmap_html(corr_df: pd.DataFrame, lag: int = 5) -> str:
    """Pearson r heatmap at a specific lag."""
    try:
        import plotly.graph_objects as go
        import plotly.offline as pyo
    except ImportError:
        return "<p><em>Plotly not available.</em></p>"

    df = corr_df[corr_df["lag_days"] == lag].copy()
    pivot = df.pivot_table(index="cameo_root", columns="commodity",
                           values="pearson_r", aggfunc="mean")

    root_labels = [
        f"{r}: {CAMEO_ROOTS.get(str(r), type('', (), {'label': str(r)})()).label[:28]}"
        for r in pivot.index
    ]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[config.TICKER_NAMES.get(c, c) for c in pivot.columns],
        y=root_labels,
        colorscale="RdBu_r",
        zmid=0, zmin=-0.4, zmax=0.4,
        colorbar=dict(title="Pearson r"),
    ))
    fig.update_layout(
        title=f"Cross-Correlation Heatmap — CAMEO vs Volatility (lag={lag} days)",
        height=700,
        margin=dict(l=320),
    )
    return pyo.plot(fig, output_type="div", include_plotlyjs=False)


def _gsi_volatility_chart_html(gsi_df: pd.DataFrame, vol_df: pd.DataFrame) -> str:
    """Dual-axis time series: GSI (left) + commodity volatilities (right)."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.offline as pyo
    except ImportError:
        return "<p><em>Plotly not available.</em></p>"

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=gsi_df.index, y=gsi_df["gsi_30d_ma"],
            name="GSI (30d MA)", line=dict(color="crimson", width=2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=gsi_df.index, y=gsi_df["gsi"],
            name="GSI (daily)", line=dict(color="salmon", width=0.5),
            opacity=0.4,
        ),
        secondary_y=False,
    )

    colours = {"CL=F": "steelblue", "GC=F": "goldenrod", "ZW=F": "seagreen"}
    for ticker in config.TICKERS:
        col = f"{ticker}_{config.PRIMARY_VOL_METRIC}"
        if col in vol_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=vol_df.index, y=vol_df[col],
                    name=f"{config.TICKER_NAMES[ticker]} σ(20d)",
                    line=dict(color=colours.get(ticker, "grey"), width=1.2),
                ),
                secondary_y=True,
            )

    fig.update_layout(
        title="Geopolitical Stress Index vs Commodity Volatility",
        xaxis_title="Date",
        height=450,
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_yaxes(title_text="GSI (stress score)", secondary_y=False)
    fig.update_yaxes(title_text="20-day rolling σ (log returns)", secondary_y=True)

    return pyo.plot(fig, output_type="div", include_plotlyjs=False)


def _rolling_corr_chart_html(rolling_corr: pd.DataFrame) -> str:
    """Rolling 90-day GSI–volatility correlation chart."""
    try:
        import plotly.graph_objects as go
        import plotly.offline as pyo
    except ImportError:
        return "<p><em>Plotly not available.</em></p>"

    fig = go.Figure()
    colours = {"CL=F": "steelblue", "GC=F": "goldenrod", "ZW=F": "seagreen"}
    for col in rolling_corr.columns:
        ticker = col.replace("gsi_vs_", "")
        fig.add_trace(go.Scatter(
            x=rolling_corr.index, y=rolling_corr[col],
            name=config.TICKER_NAMES.get(ticker, ticker),
            line=dict(color=colours.get(ticker, "grey")),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(
        title="Rolling 90-day Pearson Correlation: GSI vs Commodity Volatility",
        xaxis_title="Date", yaxis_title="Pearson r",
        yaxis=dict(range=[-1, 1]),
        height=380,
    )
    return pyo.plot(fig, output_type="div", include_plotlyjs=False)


# ── Executive summary table ───────────────────────────────────────────────────

def _top_granger_table_html(granger_df: pd.DataFrame, n: int = 5) -> str:
    """HTML table of top-N significant Granger predictors per commodity."""
    fwd = granger_df[
        (granger_df["direction"] == "cameo→vol") &
        (granger_df["significant"])
    ].copy()

    if fwd.empty:
        return "<p>No statistically significant Granger predictors found (FDR q&lt;0.05).</p>"

    html_parts = []
    for ticker in config.TICKERS:
        sub = fwd[fwd["commodity"] == ticker].nsmallest(n, "q_value_fdr")
        if sub.empty:
            html_parts.append(
                f"<h4>{config.TICKER_NAMES[ticker]}</h4>"
                "<p>No significant predictors.</p>"
            )
            continue

        sub = sub[["cameo_root", "cameo_label", "lag", "f_stat", "q_value_fdr"]].copy()
        sub.columns = ["CAMEO Code", "Category", "Lag (days)", "F-stat", "q-value (FDR)"]
        sub["F-stat"]       = sub["F-stat"].round(3)
        sub["q-value (FDR)"] = sub["q-value (FDR)"].apply(lambda v: f"{v:.4f}")
        html_parts.append(
            f"<h4>{config.TICKER_NAMES[ticker]}</h4>"
            + sub.to_html(index=False, border=0, classes="results-table")
        )

    return "\n".join(html_parts)


# ── Main report builder ───────────────────────────────────────────────────────

def generate_report(
    granger_path: Path,
    correlation_path: Path,
    gsi_path: Path,
    volatility_path: Path,
    report_path: Path,
) -> None:
    """
    Generate the self-contained HTML report.

    All charts are embedded inline; the file can be opened in any browser
    without an internet connection.
    """
    log.info("Loading result data for report…")

    granger_df = pd.read_csv(granger_path) if granger_path.exists() else pd.DataFrame()
    corr_df    = pd.read_csv(correlation_path) if correlation_path.exists() else pd.DataFrame()
    gsi_df     = load_gsi(gsi_path)
    vol_df     = pd.read_parquet(volatility_path)

    # Rolling correlation (may not exist if analysis was partial)
    rolling_corr = (
        pd.read_parquet(_ROLLING_CORR_PATH)
        if _ROLLING_CORR_PATH.exists()
        else pd.DataFrame()
    )

    # Stationarity table
    stat_table = (
        pd.read_csv(_STATIONARITY_PATH).to_html(index=False, border=0, classes="results-table")
        if _STATIONARITY_PATH.exists()
        else "<p>Stationarity report not found.</p>"
    )

    log.info("Building Plotly charts…")

    try:
        import plotly.offline as pyo
        plotlyjs = pyo.get_plotlyjs()
        plotlyjs_tag = f"<script>{plotlyjs}</script>"
    except ImportError:
        plotlyjs_tag = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'

    granger_hmap   = _granger_heatmap_html(granger_df) if not granger_df.empty else ""
    corr_hmap      = _correlation_heatmap_html(corr_df) if not corr_df.empty else ""
    gsi_chart      = _gsi_volatility_chart_html(gsi_df, vol_df)
    rolling_chart  = _rolling_corr_chart_html(rolling_corr) if not rolling_corr.empty else ""
    top_granger    = _top_granger_table_html(granger_df)  if not granger_df.empty else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Geopolitical Stress vs. Commodity Volatility Index — Results Report</title>
{plotlyjs_tag}
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1400px; margin: 0 auto; padding: 24px; color: #222; }}
  h1   {{ color: #1a1a2e; border-bottom: 3px solid #e63946; padding-bottom: 8px; }}
  h2   {{ color: #1a1a2e; margin-top: 48px; }}
  h3   {{ color: #457b9d; }}
  h4   {{ color: #333; margin: 16px 0 4px; }}
  .results-table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }}
  .results-table th {{ background: #1a1a2e; color: #fff; padding: 8px 12px; text-align: left; }}
  .results-table td {{ padding: 6px 12px; border-bottom: 1px solid #ddd; }}
  .results-table tr:hover td {{ background: #f0f4f8; }}
  .meta  {{ background: #f8f9fa; border-left: 4px solid #457b9d; padding: 12px 16px;
            border-radius: 4px; margin: 16px 0; font-size: 13px; }}
  .warn  {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px 16px;
            border-radius: 4px; margin: 16px 0; }}
  .section {{ margin-bottom: 48px; }}
</style>
</head>
<body>

<h1>Geopolitical Stress vs. Commodity Volatility Index</h1>

<div class="meta">
  <strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}<br>
  <strong>Analysis window:</strong> {config.START_DATE} → {config.END_DATE}<br>
  <strong>Commodities:</strong> Crude Oil (CL=F), Gold (GC=F), Wheat (ZW=F)<br>
  <strong>GDELT source:</strong> V2.0 master CSV files (data.gdeltproject.org)<br>
  <strong>Granger correction:</strong> Benjamini-Hochberg FDR (q &lt; {config.GRANGER_SIGNIFICANCE})
</div>

<div class="section">
<h2>1. Executive Summary — Leading Indicators</h2>
<p>CAMEO event categories that Granger-cause commodity volatility spikes
   at FDR-corrected significance q &lt; {config.GRANGER_SIGNIFICANCE}:</p>
{top_granger}
</div>

<div class="section">
<h2>2. GSI vs. Commodity Volatility</h2>
<p>Composite Geopolitical Stress Index (30-day moving average) overlaid
   with 20-day rolling log-return standard deviation for each commodity.</p>
{gsi_chart}
</div>

<div class="section">
<h2>3. Granger Causality Heatmap (lag = 5 days)</h2>
<p>Each cell shows the FDR-adjusted q-value transformed as −log₁₀(q).
   Values above 1.3 (green) indicate q &lt; 0.05.
   Cells below 1.3 (red) are not significant after FDR correction.</p>
{granger_hmap}
</div>

<div class="section">
<h2>4. Cross-Correlation Heatmap (lag = 5 days)</h2>
<p>Pearson r between CAMEO category scores (5 days earlier) and commodity volatility.</p>
{corr_hmap}
</div>

<div class="section">
<h2>5. Rolling 90-Day GSI–Volatility Correlation</h2>
<p>Time-varying Pearson correlation between the composite GSI and each commodity's
   20-day rolling volatility. Spikes above zero indicate periods where geopolitical
   stress and commodity volatility moved together.</p>
{rolling_chart}
</div>

<div class="section">
<h2>6. Stationarity Audit</h2>
<p>ADF test results for all input series. Series marked diff_order &gt; 0 were
   first-differenced before Granger tests.</p>
{stat_table}
</div>

<div class="section">
<h2>7. Methodology Notes</h2>
<div class="meta">
<strong>CAMEO Scoring:</strong><br>
CategoryScore = Σ(NumMentions × |GoldsteinScale|) / Σ NumMentions × polarity<br>
where polarity = −1 for conflictual roots (10–20), +1 for cooperative (02–08).<br><br>
<strong>Volatility Metric:</strong> {config.PRIMARY_VOL_METRIC} — 20-day rolling standard deviation of daily log returns.<br><br>
<strong>Granger Test:</strong> statsmodels.tsa.stattools.grangercausalitytests, F-test p-values,
lags tested: {config.GRANGER_MAX_LAGS} trading days.<br><br>
<strong>Multiple Testing:</strong> Benjamini-Hochberg FDR correction applied separately
within each direction (cameo→vol, vol→cameo).<br><br>
<strong>Sparse threshold:</strong> CAMEO categories with &gt;{config.GRANGER_SPARSE_CUTOFF:.0%} zero-count days
excluded from Granger (shown as NaN in heatmap).
</div>
<div class="warn">
⚠ <strong>Known limitations:</strong> (1) Yahoo Finance continuous futures series are not
back-adjusted at contract rolls — roll-date returns are masked but adjacent days
may still carry residual distortion. (2) GDELT coverage is biased toward English-language
sources (~65–70% of indexed articles). (3) Granger causality does not imply economic
causality — interpret significant results as predictive associations, not causal mechanisms.
</div>
</div>

</body>
</html>"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    log.info("HTML report written: %s", report_path)
