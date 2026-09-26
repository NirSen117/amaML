"""High-recall, deterministic, bounded candidate generation.

The blocking index is SQLite-backed so the comparison side is never expanded
to a list of Python dictionaries or a nested dict/set index.
"""
from collections.abc import Iterable, Iterator, Mapping
import heapq
import logging
import os
import sqlite3
import time
import uuid

import pandas as pd

from .normalize import digit_tokens, normalize_text, phonetic_key, tokens

LOGGER = logging.getLogger(__name__)


def _rows(records: Iterable[Mapping[str, object]] | pd.DataFrame) -> Iterator[Mapping[str, object]]:
    if isinstance(records, pd.DataFrame):
        columns = tuple(records.columns)
        for values in records.itertuples(index=False, name=None):
            yield dict(zip(columns, values))
    else:
        yield from records


def _keys(row: Mapping[str, object]) -> set[str]:
    country = normalize_text(row.get("country", ""))
    name = row.get("business_name", "")
    keys = {f"c:{country}"}
    keys.update(f"n:{country}:{token}" for token in tokens(name, True))
    keys.update(f"d:{country}:{digit}" for digit in digit_tokens(row.get("business_address", "")))
    phonetic = phonetic_key(name)
    if phonetic:
        keys.add(f"p:{country}:{phonetic}")
    return keys


def iter_candidate_lists(
    source1: Iterable[Mapping[str, object]] | pd.DataFrame,
    other: Iterable[Mapping[str, object]] | pd.DataFrame,
    max_candidates: int = 500,
) -> Iterator[tuple[str, list[str]]]:
    """Yield bounded candidate lists without retaining the comparison index.

    A temporary SQLite database is created in the current working directory
    (and removed when iteration finishes), keeping the large inverted index on
    disk while retaining deterministic input-order results.
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    db_path = f".blocking-{os.getpid()}-{uuid.uuid4().hex}.sqlite"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE key_rows (key TEXT NOT NULL, ordinal INTEGER NOT NULL, "
            "entity_id TEXT NOT NULL, PRIMARY KEY (key, ordinal)) WITHOUT ROWID"
        )
        batch: list[tuple[str, int, str]] = []
        other_count = 0
        total_other = len(other) if isinstance(other, pd.DataFrame) else None
        index_progress_step = (
            max(1, min(100_000, (total_other + 99) // 100))
            if total_other is not None else 100_000
        )
        index_started = time.monotonic()
        LOGGER.info(
            "blocking index construction started%s",
            f" for {total_other} comparison records" if total_other is not None else "",
        )
        for ordinal, row in enumerate(_rows(other)):
            entity_id = str(row["entity_id"])
            batch.extend((key, ordinal, entity_id) for key in _keys(row))
            other_count += 1
            if len(batch) >= 10000:
                connection.executemany("INSERT OR IGNORE INTO key_rows VALUES (?, ?, ?)", batch)
                batch.clear()
            if other_count % index_progress_step == 0 or (
                total_other is not None and other_count == total_other
            ):
                elapsed = max(time.monotonic() - index_started, 1e-9)
                rate = other_count / elapsed
                eta = ((total_other - other_count) / rate) if total_other else 0.0
                LOGGER.info(
                    "blocking index progress: %d/%s comparison records "
                    "(%.1f records/s, elapsed %.1fs, ETA %.1fs)",
                    other_count, total_other or "?", rate, elapsed, eta,
                )
        if batch:
            connection.executemany("INSERT OR IGNORE INTO key_rows VALUES (?, ?, ?)", batch)
        connection.commit()
        LOGGER.info("blocking index built for %d comparison records; querying candidates", other_count)

        # The primary key is clustered by (key, ordinal).  Reading a bounded,
        # already-ordered slice per key avoids SQLite's global GROUP BY/temp
        # sort.  Merging those slices is bounded by max_candidates.
        key_query = "SELECT ordinal, entity_id FROM key_rows WHERE key = ? ORDER BY ordinal LIMIT ?"
        total_source = len(source1) if isinstance(source1, pd.DataFrame) else None
        progress_step = (
            max(1, min(10_000, (total_source + 99) // 100))
            if total_source is not None
            else 10_000
        )
        started = time.monotonic()
        source_count = 0
        for source_count, left in enumerate(_rows(source1), 1):
            keys = _keys(left)
            if keys:
                streams = [
                    iter(connection.execute(key_query, (key, max_candidates)))
                    for key in keys
                ]
                merged = heapq.merge(*streams, key=lambda row: row[0])
                candidates = []
                last_ordinal = -1
                for ordinal, entity_id in merged:
                    # The same comparison row can occur under several keys.
                    if ordinal == last_ordinal:
                        continue
                    last_ordinal = ordinal
                    candidates.append(entity_id)
                    if len(candidates) >= max_candidates:
                        break
            else:
                candidates = []
            yield str(left["entity_id"]), candidates
            if source_count % progress_step == 0 or (
                total_source is not None and source_count == total_source
            ):
                elapsed = max(time.monotonic() - started, 1e-9)
                rate = source_count / elapsed
                suffix = ""
                if total_source is not None:
                    remaining = max(0, total_source - source_count)
                    suffix = f", ETA {remaining / rate:.1f}s"
                    progress = f"{source_count}/{total_source}"
                else:
                    progress = str(source_count)
                LOGGER.info(
                    "blocking candidate progress: %s source-1 records "
                    "(%.1f records/s, elapsed %.1fs%s)",
                    progress, rate, elapsed, suffix,
                )
        LOGGER.info("blocking complete: %d source-1 candidate lists", source_count)
    finally:
        connection.close()
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass


def generate_candidates(
    source1: Iterable[Mapping[str, object]] | pd.DataFrame,
    other: Iterable[Mapping[str, object]] | pd.DataFrame,
    max_candidates: int = 500,
) -> dict[str, list[str]]:
    """Return the historical mapping API, using bounded disk-backed indexing."""
    return dict(iter_candidate_lists(source1, other, max_candidates))
