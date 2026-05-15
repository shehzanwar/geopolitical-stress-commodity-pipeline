# Geopolitical Stress vs. Commodity Volatility

**Does news about wars, coups, and sanctions make oil, gold, and wheat prices swing more?**  
This project tried to answer that question rigorously — with 7 years of global event data and 480 statistical tests. The short answer: not at the daily level. The longer answer is more interesting.

---

## What This Project Does

Every 15 minutes, a system called [GDELT](https://www.gdeltproject.org/) scans news from around the world and codes every reported event using a taxonomy called **CAMEO** — a standard academic system with 20 categories ranging from "verbal cooperation" (diplomacy, praise) to "use of unconventional mass violence" (chemical weapons, atrocities). Each event is scored for intensity on a scale from −10 (most hostile) to +10 (most cooperative).

This project builds a pipeline that:

1. **Downloads and processes** ~7 years of GDELT event data (2018–2025), aggregating millions of events into daily scores for each of the 20 CAMEO categories
2. **Downloads commodity price data** for crude oil (WTI), gold, and wheat futures from Yahoo Finance
3. **Constructs a "Geopolitical Stress Index" (GSI)** — a composite daily score combining the most conflict-related event categories, weighted by media attention
4. **Measures market volatility** for each commodity using a 20-day rolling standard deviation of daily log returns
5. **Tests statistically** whether elevated geopolitical stress on Day X predicts higher commodity volatility on Day X+1, X+5, X+10, or X+20

---

## The Results

**No significant relationship was found.**

After running 480 Granger causality tests (and correcting for the fact that running 480 tests means you'd expect ~24 false positives by chance even if nothing real is happening), zero tests came back significant.

The correlation between the GSI and commodity volatility was consistently near zero:

| Relationship | Max Pearson r | Interpretation |
|---|---|---|
| GSI vs. Crude Oil volatility | ~0.05 | Negligible |
| GSI vs. Gold volatility | ~0.05 | Negligible |
| GSI vs. Wheat volatility | ~0.04 | Negligible |

For context: an r of 1.0 means perfect prediction, 0.0 means no relationship at all. Values below 0.1 are generally considered trivial.

---

## Why the Null Result Makes Sense

This might seem surprising — surely geopolitical events move commodity prices? They do. But this project was measuring something subtler: **does yesterday's stress score predict tomorrow's price swings?**

There are three reasons it doesn't:

**1. Markets react in minutes, not days.**  
Financial markets are efficient. When a missile strike happens, oil prices move within seconds of the first tweet. By the time GDELT has aggregated that event into a daily score, the market has already priced it in. Daily data is simply too slow to capture a signal that the market processes in real time.

**2. Realized volatility is backward-looking.**  
The volatility measure used here is a 20-day rolling window — it summarizes how much prices *have already moved* over the past month. It doesn't capture whether tomorrow will be volatile; it describes whether the past three weeks were. A leading signal from geopolitical stress would need a forward-looking volatility measure, like options-implied volatility (the VIX for oil, or OVX).

**3. Aggregate stress scores dilute the signal.**  
The 20 CAMEO categories, averaged across thousands of daily events worldwide, produce a relatively smooth, slow-moving series. The violent individual events that actually move markets (a pipeline explosion in Iraq, a coup in a wheat-exporting country) are buried inside global averages. Country-level or commodity-specific event filtering would likely perform better.

---

## What Might Actually Work

Based on what this project found (and didn't find), more promising approaches would include:

- **Intraday or hourly data** — matching the timescale at which markets actually react
- **Implied volatility (OVX, GVZ)** instead of realized volatility — forward-looking, priced from options markets
- **Country-filtered events** — only GDELT events in oil-producing countries for crude oil, or wheat-exporting countries for wheat
- **Event-study methodology** — measuring the market's reaction in the hours around specific high-severity events, rather than daily aggregates
- **Text-based news sentiment** — using LLM-scored sentiment from actual news articles rather than structured CAMEO codes

---

## Pipeline Architecture

```
GDELT V2 (15-min export files)          Yahoo Finance (daily OHLCV)
         |                                        |
         v                                        v
  [1. INGESTION]                          [1. INGESTION]
  gdelt_downloader.py                  commodity_loader.py
  gdelt_parser.py                              |
         |                                        |
         +------------------+--------------------+
                            |
                    [2. NLP & SCORING]
                    cameo_scorer.py  ->  cameo_scores_daily.parquet
                    stress_index.py  ->  gsi_series.parquet
                            |
               +------------+------------+
               |                         |
       [3. VOLATILITY]           [3. VOLATILITY]
       returns.py                estimators.py
               |                         |
               +------------+------------+
                            |
                   [4. ANALYSIS]
                   stationarity.py  (ADF + KPSS tests)
                   granger.py       ->  granger_results.csv
                   correlation.py   ->  correlation_matrix.csv
                   reporter.py      ->  report.html
```

Each stage writes a cached Parquet file, so re-runs skip already-completed work.

---

## Setup & Usage

### Requirements

- Python 3.10+
- ~4GB RAM for processing
- ~30GB disk space during the GDELT download (can be deleted afterward; processed data is ~3MB)

### Install

```bash
git clone https://github.com/shehzanwar/geopolitical-stress-commodity-pipeline
cd geopolitical-stress-commodity-pipeline
pip install -r requirements.txt
```

### Run

```bash
# Full pipeline (2015 to today)
python run_pipeline.py

# Custom date range
python run_pipeline.py --start 2018-01-01 --end 2025-12-31

# Re-run everything, ignoring cached files
python run_pipeline.py --force

# Run only one stage
python run_pipeline.py --stage ingestion
python run_pipeline.py --stage nlp
python run_pipeline.py --stage volatility
python run_pipeline.py --stage analysis
```

Results are written to `data/results/`:
- `granger_results.csv` — full table of 480 Granger causality tests with FDR-adjusted p-values
- `correlation_matrix.csv` — Pearson and Spearman correlations at lags 0, 1, 5, 10, 20 days
- `report.html` — visual HTML report with heatmaps and time series charts

---

## Project Structure

```
geo_stress_commodity/
├── config.py                   # All settings: dates, tickers, thresholds, paths
├── run_pipeline.py             # Orchestrator: run all 4 stages end-to-end
├── requirements.txt
│
├── ingestion/
│   ├── gdelt_downloader.py     # Downloads GDELT V2 15-min export ZIP files
│   ├── gdelt_parser.py         # Parses and aggregates events to daily Parquet
│   └── commodity_loader.py     # Fetches OHLCV from Yahoo Finance via yfinance
│
├── nlp/
│   ├── cameo_codes.py          # CAMEO taxonomy: 20 root categories + metadata
│   ├── cameo_scorer.py         # Weighted daily scores per CAMEO category
│   └── stress_index.py         # Composite GSI from conflict-weighted categories
│
├── volatility/
│   ├── returns.py              # Log returns + futures roll detection
│   └── estimators.py           # Realized vol, Parkinson, Garman-Klass, ATR
│
├── analysis/
│   ├── stationarity.py         # ADF + KPSS unit root tests
│   ├── granger.py              # 480-test Granger suite with BH-FDR correction
│   ├── correlation.py          # Cross-correlation functions at multiple lags
│   └── reporter.py             # HTML report generator
│
└── data/
    ├── raw/                    # GDELT ZIP files (large; safe to delete after run)
    ├── processed/              # Cached Parquet files (~3MB total)
    └── results/                # Final outputs: CSVs + HTML report
```

---

## Methodology Notes

**Granger Causality:** A Granger test doesn't prove that X *causes* Y. It tests whether knowing X's history improves prediction of Y's future, over and above Y's own history. It's a statistical concept of "predictive causality," not physical causality. All 480 tests were corrected for multiple comparisons using the Benjamini-Hochberg False Discovery Rate procedure.

**Stationarity:** Before running Granger tests, all time series were tested for unit roots (ADF and KPSS tests). Non-stationary series were differenced to meet the stationarity assumption required by Granger causality.

**Sparse categories:** CAMEO categories with more than 60% zero-days were flagged and excluded from Granger tests to avoid spurious results from near-constant series.

**Volatility estimators:** Four estimators were computed — 5/10/20-day realized standard deviation, Parkinson (high-low range), Garman-Klass (OHLC), and Average True Range. The primary metric for statistical tests was 20-day realized standard deviation (`rv_std_20`).

---

## Limitations

- GDELT aggregates all global events, not events specifically relevant to commodity-producing regions
- Daily aggregation loses the intraday timing of market reactions
- Realized volatility is backward-looking; options-implied volatility would be more appropriate for a leading-indicator test
- CAMEO coding is automated (machine-classified) and contains noise

---

## Author

**Shehzad Anwar** — [github.com/shehzanwar](https://github.com/shehzanwar)

*Built as an independent research project to explore quantitative relationships between geopolitical events and financial market behavior.*
