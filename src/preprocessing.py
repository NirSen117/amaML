"""Text preprocessing API."""
from entity_resolution.normalize import normalize_text, tokens, compact, digit_tokens, phonetic_key
__all__ = ["normalize_text", "tokens", "compact", "digit_tokens", "phonetic_key"]
