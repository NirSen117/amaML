"""Deterministic, language-agnostic text normalization."""
import re
import unicodedata
from typing import Iterable

_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")
_LEGAL = {"limited", "ltd", "private", "pvt", "inc", "incorporated", "llc", "corp", "corporation"}

def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _NON_WORD.sub(" ", text)
    return _SPACE.sub(" ", text).strip()

def tokens(value: object, remove_legal: bool = False) -> tuple[str, ...]:
    result = tuple(normalize_text(value).split())
    return tuple(x for x in result if not (remove_legal and x in _LEGAL))

def compact(value: object) -> str:
    return "".join(tokens(value))

def digit_tokens(value: object) -> tuple[str, ...]:
    return tuple(re.findall(r"\d+", normalize_text(value)))

def phonetic_key(value: object) -> str:
    """Small dependency-free Soundex-like key, useful for blocking only."""
    word = next(iter(tokens(value, remove_legal=True)), "")
    if not word:
        return ""
    groups = {"bfpv": "1", "cgjkqsxz": "2", "dt": "3", "l": "4", "mn": "5", "r": "6"}
    code = word[0].upper()
    previous = next((v for chars, v in groups.items() if word[0] in chars), "")
    for char in word[1:]:
        current = next((v for chars, v in groups.items() if char in chars), "")
        if current and current != previous:
            code += current
        previous = current
    return (code + "000")[:4]
