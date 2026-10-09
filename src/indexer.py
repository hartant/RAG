"""Build and save the search index."""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .chunker import chunk_file
from .models import Chunk
from .tokenizer import tokenize

EXTENSIONS = (".py", ".md")


def collect_files(raw_dir: str) -> List[str]:
    """List the .py and .md files under raw_dir, sorted."""
    root = Path(raw_dir)
    paths: List[str] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in EXTENSIONS:
            paths.append(p.as_posix())
    return sorted(paths)


def read_text(path: str) -> Optional[str]:
    """Read a file keeping original line endings; None if unreadable."""
    try:
        with open(path, encoding="utf-8", newline="") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


Postings = Dict[str, List[Tuple[int, int]]]


def build_index(
    paths: List[str], max_size: int
) -> Tuple[List[Chunk], Postings, List[int]]:
    """Chunk every file and build the inverted index."""
    chunks: List[Chunk] = []
    postings: Postings = {}
    lengths: List[int] = []
    for path in tqdm(paths, desc="Chunking"):
        text = read_text(path)
        if text is None:
            continue
        for chunk in chunk_file(path, text, max_size):
            chunk_id = len(chunks)
            chunks.append(chunk)
            tokens = tokenize(chunk.text)
            lengths.append(len(tokens))
            for word, tf in Counter(tokens).items():
                postings.setdefault(word, []).append((chunk_id, tf))
    return chunks, postings, lengths


def save_index(
    out_dir: str,
    chunks: List[Chunk],
    postings: Postings,
    lengths: List[int],
) -> None:
    """Write the index files into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {
        "chunks.json": [c.model_dump() for c in chunks],
        "postings.json": postings,
        "lengths.json": lengths,
    }
    for name, content in data.items():
        with open(out / name, "w", encoding="utf-8") as f:
            json.dump(content, f)


def load_index(
    in_dir: str,
) -> Tuple[List[Chunk], Postings, List[int]]:
    """Read the index files from in_dir."""
    base = Path(in_dir)
    with open(base / "chunks.json", encoding="utf-8") as f:
        chunks = [Chunk(**c) for c in json.load(f)]
    with open(base / "postings.json", encoding="utf-8") as f:
        raw = json.load(f)
    postings: Postings = {
        word: [(int(i), int(tf)) for i, tf in plist] for word, plist in raw.items()
    }
    with open(base / "lengths.json", encoding="utf-8") as f:
        lengths: List[int] = json.load(f)
    return chunks, postings, lengths
