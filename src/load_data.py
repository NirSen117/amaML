"""Data loading API; all reads are explicit TSV reads."""
from entity_resolution.io import read_records, read_ground_truth
__all__ = ["read_records", "read_ground_truth"]
