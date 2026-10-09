"""BM25 retrieval over the saved index."""

import math
from collections import Counter
from typing import Dict, List

from .indexer import Postings
from .models import Chunk, MinimalSource
from .tokenizer import tokenize

K1 = 1.5
B = 0.75


class Retriever:
    """Rank chunks for a query using BM25."""

    def __init__(
        self, chunks: List[Chunk], postings: Postings, lengths: List[int]
    ) -> None:
        """Store the index and precompute the average length."""
        self.chunks = chunks
        self.postings = postings
        self.lengths = lengths
        total = sum(lengths)
        self.avg_len = total / len(lengths) if lengths else 0.0

    def idf(self, word: str) -> float:
        """Inverse document frequency of a word."""
        n = len(self.postings.get(word, []))
        big_n = len(self.chunks)
        return math.log(1 + (big_n - n + 0.5) / (n + 0.5))

    def search(self, query: str, k: int) -> List[MinimalSource]:
        """Return the top-k sources for a query."""
        if k <= 0 or not self.chunks:
            return []
        scores: Dict[int, float] = {}
        for word, qtf in Counter(tokenize(query)).items():
            plist = self.postings.get(word)
            if not plist:
                continue
            idf = self.idf(word)
            for chunk_id, tf in plist:
                norm = K1 * (1 - B + B * self.lengths[chunk_id] / self.avg_len)
                part = tf * (K1 + 1) / (tf + norm)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * part
        best = sorted(scores, key=lambda i: scores[i], reverse=True)[:k]
        return [
            MinimalSource(
                file_path=self.chunks[i].file_path,
                first_character_index=self.chunks[i].first_character_index,
                last_character_index=self.chunks[i].last_character_index,
            )
            for i in best
        ]
