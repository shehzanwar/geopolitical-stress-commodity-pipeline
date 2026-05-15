"""
run_pipeline.py — End-to-end orchestrator for the Geopolitical Stress vs. Commodity
Volatility Index pipeline.

Usage:
    python run_pipeline.py                          # Full run, default date range
    python run_pipeline.py --start 2020-01-01       # Custom start date
    python run_pipeline.py --end   2023-12-31       # Custom end date
    python run_pipeline.py --force                  # Re-run all stages (ignore cache)
    python run_pipeline.py --stage ingestion        # Run only one stage
    python run_pipeline.py --stage nlp
    python run_pipeline.py --stage volatility
    python run_pipeline.py --stage analysis

Stages run in dependency order:
    1. ingestion  → gdelt_daily.parquet + commodity_ohlcv.parquet
    2. nlp        → cameo_scores_daily.parquet + gsi_series.parquet
    3. volatility → volatility_daily.parquet
    4. analysis   → granger_results.csv + correlation_matrix.csv + report.html
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import config
from ingestion.gdelt_downloader import download_gdelt_range
from ingestion.gdelt_parser import parse_gdelt_to_daily_parquet
from ingestion.commodity_loader import load_commodity_ohlcv
from nlp.cameo_scorer import build_cameo_scores
from nlp.stress_index import build_gsi
from volatility.returns import compute_log_returns
from volatility.estimators import compute_all_volatility
from analysis.stationarity import run_stationarity_checks
from analysis.granger import run_granger_suite
from analysis.correlation import run_correlation_analysis
from analysis.reporter import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.ROOT_DIR / "pipeline.log"),
    ],
)
log = logging.getLogger(__name__)


# ── Stage helpers ─────────────────────────────────────────────────────────────

def _needs_run(path: Path, force: bool) -> bool:
    """Return True if the stage output is missing or --force is set."""
    if force:
        return True
    if not path.exists():
        log.info("Cache miss: %s not found — will run stage.", path.name)
        return True
    log.info("Cache hit:  %s exists — skipping stage (use --force to override).", path.name)
    return False


def stage_ingestion(start: str, end: str, force: bool) -> None:
    log.info("=" * 60)
    log.info("STAGE 1: INGESTION")
    log.info("=" * 60)

    # ── GDELT ─────────────────────────────────────────────────────
    if _needs_run(config.GDELT_DAILY_PATH, force):
        t0 = time.time()
        log.info("Downloading GDELT V2 export files: %s → %s", start, end)
        raw_files = download_gdelt_range(start, end)
        log.info("Downloaded %d GDELT file batches. Parsing to daily Parquet…", len(raw_files))
        parse_gdelt_to_daily_parquet(raw_files, config.GDELT_DAILY_PATH)
        log.info("GDELT daily Parquet written in %.1fs", time.time() - t0)

    # ── Commodities ───────────────────────────────────────────────
    if _needs_run(config.COMMODITY_OHLCV_PATH, force):
        t0 = time.time()
        log.info("Fetching commodity OHLCV: %s", config.TICKERS)
        load_commodity_ohlcv(config.TICKERS, start, end, config.COMMODITY_OHLCV_PATH)
        log.info("Commodity OHLCV written in %.1fs", time.time() - t0)


def stage_nlp(force: bool) -> None:
    log.info("=" * 60)
    log.info("STAGE 2: NLP & EVENT WEIGHTING")
    log.info("=" * 60)

    if _needs_run(config.CAMEO_SCORES_PATH, force):
        t0 = time.time()
        log.info("Building per-category CAMEO scores from %s", config.GDELT_DAILY_PATH.name)
        build_cameo_scores(config.GDELT_DAILY_PATH, config.CAMEO_SCORES_PATH)
        log.info("CAMEO scores written in %.1fs", time.time() - t0)

    if _needs_run(config.GSI_PATH, force):
        t0 = time.time()
        log.info("Computing composite Geopolitical Stress Index (GSI)…")
        build_gsi(config.CAMEO_SCORES_PATH, config.GSI_PATH)
        log.info("GSI written in %.1fs", time.time() - t0)


def stage_volatility(force: bool) -> None:
    log.info("=" * 60)
    log.info("STAGE 3: VOLATILITY CALCULATION")
    log.info("=" * 60)

    if _needs_run(config.VOLATILITY_PATH, force):
        t0 = time.time()
        log.info("Computing log returns and volatility estimators…")
        returns_df = compute_log_returns(config.COMMODITY_OHLCV_PATH)
        compute_all_volatility(returns_df, config.VOLATILITY_PATH)
        log.info("Volatility Parquet written in %.1fs", time.time() - t0)


def stage_analysis(force: bool) -> None:
    log.info("=" * 60)
    log.info("STAGE 4: STATISTICAL ANALYSIS")
    log.info("=" * 60)

    # Stationarity pre-check (always run — lightweight and informative)
    log.info("Running stationarity checks (ADF + KPSS)…")
    stationary_cameo, stationary_vol = run_stationarity_checks(
        config.CAMEO_SCORES_PATH, config.VOLATILITY_PATH
    )

    if _needs_run(config.GRANGER_RESULTS_PATH, force):
        t0 = time.time()
        log.info("Running Granger causality suite (240 tests)…")
        run_granger_suite(stationary_cameo, stationary_vol, config.GRANGER_RESULTS_PATH)
        log.info("Granger results written in %.1fs", time.time() - t0)

    if _needs_run(config.CORRELATION_PATH, force):
        t0 = time.time()
        log.info("Running cross-correlation analysis…")
        run_correlation_analysis(
            stationary_cameo, stationary_vol,
            config.GSI_PATH, config.CORRELATION_PATH
        )
        log.info("Correlation matrix written in %.1fs", time.time() - t0)

    log.info("Generating HTML report…")
    generate_report(
        granger_path=config.GRANGER_RESULTS_PATH,
        correlation_path=config.CORRELATION_PATH,
        gsi_path=config.GSI_PATH,
        volatility_path=config.VOLATILITY_PATH,
        report_path=config.REPORT_PATH,
    )
    log.info("Report written to: %s", config.REPORT_PATH)


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Geopolitical Stress vs. Commodity Volatility Index pipeline"
    )
    parser.add_argument(
        "--start", default=config.START_DATE,
        help=f"Analysis start date YYYY-MM-DD (default: {config.START_DATE})"
    )
    parser.add_argument(
        "--end", default=config.END_DATE,
        help=f"Analysis end date YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore cached Parquet files and re-run all stages"
    )
    parser.add_argument(
        "--stage",
        choices=["ingestion", "nlp", "volatility", "analysis"],
        default=None,
        help="Run only a specific stage (default: run all four in order)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log.info("Pipeline start  |  %s to %s  |  force=%s  |  stage=%s",
             args.start, args.end, args.force, args.stage or "all")

    t_total = time.time()

    stage_map = {
        "ingestion": lambda: stage_ingestion(args.start, args.end, args.force),
        "nlp":       lambda: stage_nlp(args.force),
        "volatility": lambda: stage_volatility(args.force),
        "analysis":  lambda: stage_analysis(args.force),
    }

    if args.stage:
        stage_map[args.stage]()
    else:
        for stage_fn in stage_map.values():
            stage_fn()

    log.info("Pipeline complete in %.1fs", time.time() - t_total)
    log.info("Results: %s", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
