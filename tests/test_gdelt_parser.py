"""
tests/test_gdelt_parser.py

Unit tests for ingestion/gdelt_parser.py using synthetic in-memory data.
No live GDELT downloads are required — all tests use fabricated DataFrames
that mimic the structure of real GDELT V2 export files.
"""

import io
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Module under test ─────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.gdelt_parser import (
    _read_zip,
    _aggregate_day,
    _COL_NAMES,
    _USECOLS,
    _VALID_ROOTS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_raw_row(
    event_id: int = 1,
    sqldate: int = 20220224,
    event_root: str = "18",
    goldstein: float = -8.0,
    num_mentions: int = 10,
    num_sources: int = 3,
    num_articles: int = 5,
    avg_tone: float = -5.0,
) -> list:
    """Build a 61-element list mimicking one GDELT V2 export row."""
    row = [""] * 61
    row[0]  = str(event_id)
    row[1]  = str(sqldate)
    row[26] = "181"          # EventCode
    row[27] = "18"           # EventBaseCode
    row[28] = event_root     # EventRootCode
    row[30] = str(goldstein)
    row[31] = str(num_mentions)
    row[32] = str(num_sources)
    row[33] = str(num_articles)
    row[34] = str(avg_tone)
    return row


def _make_tsv_bytes(rows: list[list]) -> bytes:
    """Convert a list of rows into tab-delimited bytes (no header)."""
    lines = ["\t".join(r) for r in rows]
    return "\n".join(lines).encode("utf-8")


def _make_zip_file(rows: list[list], tmp_dir: Path) -> Path:
    """Write a fake GDELT export zip to tmp_dir and return its path."""
    csv_bytes = _make_tsv_bytes(rows)
    zip_path  = tmp_dir / "20220224000000.export.CSV.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("20220224000000.export.CSV", csv_bytes)
    return zip_path


# ── Tests for _read_zip ───────────────────────────────────────────────────────

class TestReadZip:
    def test_reads_valid_zip(self, tmp_path):
        rows = [_make_raw_row(event_id=i, num_articles=5) for i in range(1, 4)]
        zpath = _make_zip_file(rows, tmp_path)
        df = _read_zip(zpath)
        assert df is not None
        assert len(df) == 3
        assert "GlobalEventID" in df.columns
        assert "EventRootCode" in df.columns

    def test_returns_none_for_bad_zip(self, tmp_path):
        bad_zip = tmp_path / "bad.zip"
        bad_zip.write_bytes(b"not a zip file")
        result = _read_zip(bad_zip)
        assert result is None

    def test_returns_none_for_empty_zip(self, tmp_path):
        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("empty.CSV", "")
        result = _read_zip(zip_path)
        # Empty CSV → empty DataFrame → _read_zip returns None
        assert result is None or (result is not None and result.empty)

    def test_eventcode_stored_as_string(self, tmp_path):
        """EventCode must remain a string (zero-padding preservation)."""
        rows = [_make_raw_row(event_id=1, num_articles=5)]
        rows[0][26] = "014"   # Zero-padded code
        zpath = _make_zip_file(rows, tmp_path)
        df = _read_zip(zpath)
        assert df is not None
        assert df["EventCode"].iloc[0] == "014"


# ── Tests for _aggregate_day ──────────────────────────────────────────────────

class TestAggregateDay:
    def _make_df(self, rows: list[dict]) -> pd.DataFrame:
        """Helper: build a parsed DataFrame from row dicts."""
        data = {col: [] for col in _COL_NAMES.values()}
        for r in rows:
            for key, val in r.items():
                data[key].append(val)
        # Fill missing columns
        for col in _COL_NAMES.values():
            if col not in data:
                data[col] = [None] * len(rows)
        return pd.DataFrame(data)

    def test_deduplication_by_event_id(self):
        """Same GlobalEventID in two rows should count as one event."""
        rows = [
            {"GlobalEventID": 1, "SQLDATE": 20220224, "EventRootCode": "18",
             "GoldsteinScale": -8.0, "NumMentions": 10, "NumSources": 3,
             "NumArticles": 5, "AvgTone": -5.0, "EventCode": "181", "EventBaseCode": "18"},
            {"GlobalEventID": 1, "SQLDATE": 20220224, "EventRootCode": "18",
             "GoldsteinScale": -8.0, "NumMentions": 10, "NumSources": 3,
             "NumArticles": 5, "AvgTone": -5.0, "EventCode": "181", "EventBaseCode": "18"},
        ]
        df = self._make_df(rows)
        agg = _aggregate_day(df)
        assert len(agg) == 1
        assert agg["n_events"].iloc[0] == 1

    def test_quality_filter_removes_low_article_events(self):
        """Events with NumArticles < GDELT_MIN_ARTICLES (3) must be dropped."""
        rows = [
            {"GlobalEventID": 1, "SQLDATE": 20220224, "EventRootCode": "13",
             "GoldsteinScale": -6.0, "NumMentions": 5, "NumSources": 2,
             "NumArticles": 1,   # below threshold
             "AvgTone": -3.0, "EventCode": "13", "EventBaseCode": "13"},
            {"GlobalEventID": 2, "SQLDATE": 20220224, "EventRootCode": "19",
             "GoldsteinScale": -9.0, "NumMentions": 20, "NumSources": 8,
             "NumArticles": 10,  # above threshold
             "AvgTone": -8.0, "EventCode": "19", "EventBaseCode": "19"},
        ]
        df = self._make_df(rows)
        agg = _aggregate_day(df)
        # Only root 19 should survive
        assert len(agg) == 1
        assert agg["event_root"].iloc[0] == "19"

    def test_groups_by_root_code(self):
        """Events with different roots should produce separate rows."""
        rows = [
            {"GlobalEventID": 1, "SQLDATE": 20220224, "EventRootCode": "14",
             "GoldsteinScale": -5.0, "NumMentions": 8, "NumSources": 3,
             "NumArticles": 4, "AvgTone": -4.0, "EventCode": "14", "EventBaseCode": "14"},
            {"GlobalEventID": 2, "SQLDATE": 20220224, "EventRootCode": "18",
             "GoldsteinScale": -8.0, "NumMentions": 12, "NumSources": 5,
             "NumArticles": 6, "AvgTone": -6.0, "EventCode": "181", "EventBaseCode": "18"},
        ]
        df = self._make_df(rows)
        agg = _aggregate_day(df)
        assert len(agg) == 2
        assert set(agg["event_root"]) == {"14", "18"}

    def test_invalid_root_codes_are_dropped(self):
        """Root codes outside 01–20 must be excluded."""
        rows = [
            {"GlobalEventID": 1, "SQLDATE": 20220224, "EventRootCode": "99",
             "GoldsteinScale": 0.0, "NumMentions": 5, "NumSources": 2,
             "NumArticles": 3, "AvgTone": 0.0, "EventCode": "99", "EventBaseCode": "99"},
        ]
        df = self._make_df(rows)
        agg = _aggregate_day(df)
        assert agg.empty

    def test_weighted_goldstein_computation(self):
        """
        Two events in the same root:
        Event A: Goldstein=-4, NumMentions=10  → contribution: -40
        Event B: Goldstein=-8, NumMentions=10  → contribution: -80
        sum_goldstein should be -120, n_mentions=20.
        """
        rows = [
            {"GlobalEventID": 1, "SQLDATE": 20220224, "EventRootCode": "18",
             "GoldsteinScale": -4.0, "NumMentions": 10, "NumSources": 2,
             "NumArticles": 5, "AvgTone": -2.0, "EventCode": "18", "EventBaseCode": "18"},
            {"GlobalEventID": 2, "SQLDATE": 20220224, "EventRootCode": "18",
             "GoldsteinScale": -8.0, "NumMentions": 10, "NumSources": 3,
             "NumArticles": 5, "AvgTone": -6.0, "EventCode": "18", "EventBaseCode": "18"},
        ]
        df = self._make_df(rows)
        agg = _aggregate_day(df)
        assert len(agg) == 1
        assert agg["n_mentions"].iloc[0] == 20
        assert abs(agg["sum_goldstein"].iloc[0] - (-120.0)) < 0.01

    def test_output_has_required_columns(self):
        rows = [
            {"GlobalEventID": 1, "SQLDATE": 20220224, "EventRootCode": "06",
             "GoldsteinScale": 4.0, "NumMentions": 5, "NumSources": 2,
             "NumArticles": 3, "AvgTone": 2.0, "EventCode": "06", "EventBaseCode": "06"},
        ]
        df = self._make_df(rows)
        agg = _aggregate_day(df)
        for col in ["date", "event_root", "n_events", "n_mentions",
                    "n_articles", "sum_goldstein", "sum_tone"]:
            assert col in agg.columns, f"Missing column: {col}"
