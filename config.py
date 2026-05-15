"""
config.py — Central configuration for the Geopolitical Stress vs. Commodity Volatility pipeline.

All modules import from here. Edit this file to change date ranges, tickers,
thresholds, and file paths without touching business logic.
"""

from pathlib import Path
from datetime import date

# ── Directory layout ──────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent
DATA_DIR     = ROOT_DIR / "data"
RAW_DIR      = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR  = DATA_DIR / "results"

for _d in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Analysis window ───────────────────────────────────────────────────────────
START_DATE = "2015-02-19"                       # GDELT V2.0 launch date
END_DATE   = date.today().strftime("%Y-%m-%d")  # Inclusive upper bound

# ── Commodity futures tickers ─────────────────────────────────────────────────
TICKERS = ["CL=F", "GC=F", "ZW=F"]
TICKER_NAMES = {
    "CL=F": "Crude Oil (WTI)",
    "GC=F": "Gold",
    "ZW=F": "Wheat",
}

# ── GDELT data source ─────────────────────────────────────────────────────────
GDELT_MASTER_URL       = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
GDELT_MIN_ARTICLES     = 3      # Discard events with fewer article mentions (quality filter)
GDELT_MAX_WORKERS      = 4      # Concurrent download threads
GDELT_REQUEST_TIMEOUT  = 30     # HTTP timeout in seconds
GDELT_RETRY_ATTEMPTS   = 3      # Max retries per file
GDELT_RETRY_BACKOFF    = 2.0    # Exponential backoff multiplier (seconds × 2^attempt)

# ── Volatility estimator parameters ──────────────────────────────────────────
VOL_WINDOWS              = [5, 10, 20]   # Rolling window sizes (trading days)
ATR_WINDOW               = 14            # ATR standard window
ROLL_DETECTION_THRESHOLD = 0.03          # Abs log-return > 3% flags a probable roll date

# ── Granger causality ─────────────────────────────────────────────────────────
GRANGER_MAX_LAGS      = [1, 5, 10, 20]  # Lag orders tested (trading days)
GRANGER_SIGNIFICANCE  = 0.05            # Alpha threshold (applied to FDR-adjusted q-values)
GRANGER_SPARSE_CUTOFF = 0.60            # Skip CAMEO categories with >60% zero-count days

# ── Primary volatility metric used in Granger tests ──────────────────────────
PRIMARY_VOL_METRIC = "rv_std_20"        # Column suffix in volatility_daily.parquet

# ── Processed data cache paths ────────────────────────────────────────────────
GDELT_DAILY_PATH    = PROCESSED_DIR / "gdelt_daily.parquet"
COMMODITY_OHLCV_PATH = PROCESSED_DIR / "commodity_ohlcv.parquet"
CAMEO_SCORES_PATH   = PROCESSED_DIR / "cameo_scores_daily.parquet"
GSI_PATH            = PROCESSED_DIR / "gsi_series.parquet"
VOLATILITY_PATH     = PROCESSED_DIR / "volatility_daily.parquet"

# ── Results output paths ──────────────────────────────────────────────────────
GRANGER_RESULTS_PATH = RESULTS_DIR / "granger_results.csv"
CORRELATION_PATH     = RESULTS_DIR / "correlation_matrix.csv"
REPORT_PATH          = RESULTS_DIR / "report.html"

# ── Minimum data coverage check ───────────────────────────────────────────────
MIN_COMMODITY_COVERAGE = 0.90   # Warn if any ticker has < 90% of expected trading days
