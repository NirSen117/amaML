"""Strict, path-oriented TSV input and submission output helpers."""
from pathlib import Path
from typing import Iterable
import logging
import pandas as pd

LOGGER = logging.getLogger(__name__)
REQUIRED_RECORD_COLUMNS = ("entity_id", "business_name", "business_address", "country")

def read_records(path: Path) -> pd.DataFrame:
    """Read and validate a record TSV; no implicit dataset discovery is performed."""
    LOGGER.info("loading records from %s", path)
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(REQUIRED_RECORD_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    result = frame.loc[:, list(REQUIRED_RECORD_COLUMNS)].copy()
    LOGGER.info("loaded %d records from %s", len(result), path)
    return result

def read_ground_truth(path: Path) -> pd.DataFrame:
    LOGGER.info("loading ground truth from %s", path)
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    expected = {"source1_entity_id", "matched_entity_ids"}
    if not expected.issubset(frame.columns):
        raise ValueError(f"{path} must contain {sorted(expected)}")
    result = frame.loc[:, ["source1_entity_id", "matched_entity_ids"]].copy()
    LOGGER.info("loaded %d ground-truth rows", len(result))
    return result

def write_submission(path: Path, rows: Iterable[dict[str, str]]) -> None:
    LOGGER.info("saving matching results to %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(
        path, sep="\t", index=False, lineterminator="\n"
    )

def write_candidates(path: Path, rows: Iterable[dict[str, str]]) -> None:
    LOGGER.info("saving candidate pairs to %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"]).to_csv(
        path, sep="\t", index=False, lineterminator="\n"
    )
