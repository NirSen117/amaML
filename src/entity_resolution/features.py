"""Pairwise string and field agreement features."""
from difflib import SequenceMatcher
import re
import time
from typing import Mapping
from .normalize import normalize_text, tokens, digit_tokens, phonetic_key
import logging

LOGGER = logging.getLogger(__name__)

FEATURE_NAMES = ("name_ratio", "name_token_jaccard", "name_phonetic", "address_ratio",
                 "address_token_jaccard", "pin_overlap", "country_equal", "exact_field_count")

def _jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    a, b = set(left), set(right)
    return 1.0 if not a and not b else len(a & b) / max(1, len(a | b))

def pair_features(left: Mapping[str, object], right: Mapping[str, object]) -> dict[str, float]:
    ln, rn = normalize_text(left.get("business_name", "")), normalize_text(right.get("business_name", ""))
    la, ra = normalize_text(left.get("business_address", "")), normalize_text(right.get("business_address", ""))
    lp, rp = set(digit_tokens(la)), set(digit_tokens(ra))
    lphonetic, rphonetic = phonetic_key(ln), phonetic_key(rn)
    country_equal = normalize_text(left.get("country", "")) == normalize_text(right.get("country", ""))
    return {
        "name_ratio": SequenceMatcher(None, ln, rn).ratio(),
        "name_token_jaccard": _jaccard(tokens(ln, True), tokens(rn, True)),
        "name_phonetic": float(bool(lphonetic) and lphonetic == rphonetic),
        "address_ratio": SequenceMatcher(None, la, ra).ratio(),
        "address_token_jaccard": _jaccard(tokens(la), tokens(ra)),
        "pin_overlap": float(bool(lp & rp)),
        "country_equal": float(country_equal),
        "exact_field_count": float(ln == rn) + float(la == ra),
    }

def feature_matrix(pairs: list[tuple[Mapping[str, object], Mapping[str, object]]]):
    import numpy as np
    LOGGER.info("generating features for %d pairs", len(pairs))
    matrix = []
    progress_step = max(1, min(10_000, (len(pairs) + 99) // 100))
    started = time.monotonic()
    for position, (left, right) in enumerate(pairs, 1):
        values = pair_features(left, right)
        matrix.append([values[name] for name in FEATURE_NAMES])
        if position == len(pairs) or position % progress_step == 0:
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = position / elapsed
            LOGGER.info(
                "feature generation progress: %d/%d pairs (%.1f pairs/s, "
                "elapsed %.1fs, ETA %.1fs)",
                position, len(pairs), rate, elapsed,
                max(0, len(pairs) - position) / rate,
            )
    LOGGER.info("feature generation complete: %d pairs", len(matrix))
    return np.asarray(matrix, dtype=float)
