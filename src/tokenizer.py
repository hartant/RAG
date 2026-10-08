"""Turn text into lowercase tokens for indexing and search."""

import re
from typing import List

WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """Split text into lowercase tokens, keeping identifiers and parts."""
    tokens: List[str] = []
    for word in WORD_RE.findall(text):
        low = word.lower()
        tokens.append(low)
        parts = [p for p in low.split("_") if p]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens
