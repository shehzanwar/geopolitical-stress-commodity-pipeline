"""
nlp/cameo_codes.py

Static reference data for the 20 CAMEO root event codes used throughout the
Geopolitical Stress vs. Commodity Volatility pipeline.

Each root code is mapped to:
    label    : human-readable category name
    polarity : +1 (cooperative) | −1 (conflictual) | 0 (neutral)

The polarity is used by cameo_scorer.py to sign the Goldstein-weighted score
so that conflictual events contribute positively to geopolitical stress and
cooperative events reduce it.

Source: CAMEO Conflict and Mediation Event Observations Codebook v1.1b3
        http://data.gdeltproject.org/documentation/CAMEO.Manual.1.1b3.pdf
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class CAMEORoot:
    code: str       # Zero-padded 2-digit string, e.g. "01"
    label: str
    polarity: int   # +1 cooperative, -1 conflictual, 0 neutral


# The full taxonomy, codes 01–20
CAMEO_ROOTS: Dict[str, CAMEORoot] = {
    "01": CAMEORoot("01", "Make Public Statement",         polarity=0),
    "02": CAMEORoot("02", "Appeal",                        polarity=+1),
    "03": CAMEORoot("03", "Agree",                         polarity=+1),
    "04": CAMEORoot("04", "Consult",                       polarity=+1),
    "05": CAMEORoot("05", "Support Diplomatically",        polarity=+1),
    "06": CAMEORoot("06", "Cooperate Economically/Militarily", polarity=+1),
    "07": CAMEORoot("07", "Provide Aid",                   polarity=+1),
    "08": CAMEORoot("08", "Yield / Concede",               polarity=+1),
    "09": CAMEORoot("09", "Investigate",                   polarity=0),
    "10": CAMEORoot("10", "Demand",                        polarity=-1),
    "11": CAMEORoot("11", "Disapprove",                    polarity=-1),
    "12": CAMEORoot("12", "Reject",                        polarity=-1),
    "13": CAMEORoot("13", "Threaten",                      polarity=-1),
    "14": CAMEORoot("14", "Protest",                       polarity=-1),
    "15": CAMEORoot("15", "Exhibit Military Posture",      polarity=-1),
    "16": CAMEORoot("16", "Reduce Relations",              polarity=-1),
    "17": CAMEORoot("17", "Coerce",                        polarity=-1),
    "18": CAMEORoot("18", "Assault",                       polarity=-1),
    "19": CAMEORoot("19", "Fight",                         polarity=-1),
    "20": CAMEORoot("20", "Use Unconventional Mass Violence", polarity=-1),
}

# Convenience lookups
ALL_ROOTS   = list(CAMEO_ROOTS.keys())                         # ["01", …, "20"]
CONFLICT_ROOTS = [k for k, v in CAMEO_ROOTS.items() if v.polarity == -1]   # 10–20
COOP_ROOTS     = [k for k, v in CAMEO_ROOTS.items() if v.polarity == +1]   # 02–08
NEUTRAL_ROOTS  = [k for k, v in CAMEO_ROOTS.items() if v.polarity == 0]    # 01, 09

# Column name helpers used by cameo_scorer and downstream modules
def score_col(root: str)  -> str: return f"CAM_{root}"
def count_col(root: str)  -> str: return f"count_{root}"
def tone_col(root: str)   -> str: return f"tone_{root}"
