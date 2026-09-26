"""Similarity primitives used by feature generation."""
from difflib import SequenceMatcher
from entity_resolution.features import _jaccard
def sequence_similarity(left: object, right: object) -> float:
    return SequenceMatcher(None, str(left), str(right)).ratio()
def token_jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    return _jaccard(left, right)
__all__ = ["sequence_similarity", "token_jaccard"]
