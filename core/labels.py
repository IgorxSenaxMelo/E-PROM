"""Shared vocabulary for E-PROM.

Canonical (internal) labels are English. Portuguese spellings used by older
workbooks are accepted everywhere and mapped to the canonical labels.
"""

from __future__ import annotations
import re
import unicodedata

import numpy as np


# --------------------------------------------------------------------------
# Text matching
# --------------------------------------------------------------------------

def match_key(value) -> str:
    """Case-, accent-, separator- and whitespace-insensitive comparison key."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value).strip().casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[-\u2013\u2014_/]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Sheets that never contain decision-maker data.
SUPPORT_SHEET_KEYS = {
    "guide", "guia", "_guide", "_lists", "lists", "instructions",
    "instrucoes", "orientacoes", "notes", "readme",
}


def is_support_sheet(title) -> bool:
    return match_key(title) in SUPPORT_SHEET_KEYS


# --------------------------------------------------------------------------
# Data type
# --------------------------------------------------------------------------

ORDINAL = "Ordinal"
CONTINUOUS = "Continuous"

_TYPE_ALIASES = {
    "ordinal": ORDINAL,
    "continuous": CONTINUOUS,
    "continuo": CONTINUOUS,
}


def normalize_type(value) -> str:
    key = match_key(value)
    if key not in _TYPE_ALIASES:
        raise ValueError(
            f"Unrecognized criterion type: {value!r}. Use Ordinal or Continuous."
        )
    return _TYPE_ALIASES[key]


# --------------------------------------------------------------------------
# Precision
# --------------------------------------------------------------------------

HIGH, MEDIUM, LOW = "High", "Medium", "Low"
PRECISION_LEVELS = (HIGH, MEDIUM, LOW)

_PRECISION_ALIASES = {
    "high": HIGH, "alta": HIGH,
    "medium": MEDIUM, "media": MEDIUM,
    "low": LOW, "baixa": LOW,
}


def normalize_precision(value, default: str = HIGH) -> str:
    """Return High / Medium / Low. Empty values fall back to ``default``."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    text = str(value).strip()
    if not text:
        return default
    key = match_key(text)
    if key not in _PRECISION_ALIASES:
        raise ValueError(
            f"Unrecognized precision {value!r}. Use High, Medium or Low "
            "(Alta, Média, Baixa are also accepted)."
        )
    return _PRECISION_ALIASES[key]


# --------------------------------------------------------------------------
# FAHP-Express linguistic scale (value = how many times the reference
# criterion is more important than the compared criterion)
# --------------------------------------------------------------------------

LINGUISTIC_SCALE = {
    "equal importance": 1,
    "weakly more important": 2,
    "moderately more important": 3,
    "moderately to strongly more important": 4,
    "strongly more important": 5,
    "strongly to very strongly more important": 6,
    "very strongly more important": 7,
    "very strongly to extremely more important": 8,
    "extremely more important": 9,
}

_PT_LINGUISTIC_SCALE = {
    "igual importancia": 1,
    "fracamente mais importante": 2,
    "moderadamente mais importante": 3,
    "moderadamente a fortemente mais importante": 4,
    "fortemente mais importante": 5,
    "fortemente a muito fortemente mais importante": 6,
    "muito fortemente mais importante": 7,
    "muito fortemente a extremamente mais importante": 8,
    "extremamente mais importante": 9,
}

_LINGUISTIC_BY_KEY = {
    **{match_key(k): v for k, v in LINGUISTIC_SCALE.items()},
    **_PT_LINGUISTIC_SCALE,
}

LINGUISTIC_TERMS_BY_VALUE = {v: k for k, v in LINGUISTIC_SCALE.items()}


def parse_importance(value) -> float:
    """Convert a linguistic term, a number or a numeric string to a float."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("Empty value in the FAHP-Express comparison row.")
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("NaN/inf value in the FAHP-Express comparison row.")
        return number
    text = str(value).strip()
    key = match_key(text)
    if key in _LINGUISTIC_BY_KEY:
        return float(_LINGUISTIC_BY_KEY[key])
    try:
        number = float(text.replace(",", "."))
    except ValueError:
        raise ValueError(
            f"Unrecognized FAHP-Express comparison value: {value!r}."
        ) from None
    if not np.isfinite(number):
        raise ValueError("NaN/inf value in the FAHP-Express comparison row.")
    return number
