"""Split source files into chunks (Python code and Markdown)."""

import ast
from typing import List, Tuple

from .models import Chunk

Span = Tuple[int, int]


def line_offsets(text: str) -> List[int]:
    """Return the character offset where each line starts."""
    offsets = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            offsets.append(i + 1)
    return offsets


def pack(text: str, cuts: List[int], max_size: int) -> List[Span]:
    """Group pieces delimited by `cuts` into spans <= max_size."""
    cuts = sorted({c for c in cuts if 0 < c < len(text)})
    points = cuts + [len(text)]
    spans: List[Span] = []
    start = 0
    last_ok = 0
    for p in points:
        if p - start > max_size:
            if last_ok > start:
                spans.append((start, last_ok))
                start = last_ok
            while p - start > max_size:
                spans.append((start, start + max_size))
                start += max_size
        last_ok = p
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def python_cuts(text: str) -> List[int]:
    """Offsets of every function/class start in a Python file."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return line_offsets(text)
    offsets = line_offsets(text)
    cuts: List[int] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first_line = node.lineno
            for deco in node.decorator_list:
                first_line = min(first_line, deco.lineno)
            cuts.append(offsets[first_line - 1])
    return cuts


def markdown_header_cuts(text: str) -> List[int]:
    """Offsets of lines starting with '#', ignoring code fences."""
    cuts: List[int] = []
    in_fence = False
    pos = 0
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and line.startswith("#"):
            cuts.append(pos)
        pos += len(line) + 1
    return cuts


def blank_line_cuts(text: str) -> List[int]:
    """Offsets just after each blank line (paragraph breaks)."""
    cuts: List[int] = []
    idx = text.find("\n\n")
    while idx != -1:
        cuts.append(idx + 2)
        idx = text.find("\n\n", idx + 2)
    return cuts


def markdown_spans(text: str, max_size: int) -> List[Span]:
    """Cut on headers first, then on paragraphs for oversized parts."""
    spans: List[Span] = []
    for start, end in pack(text, markdown_header_cuts(text), max_size):
        part = text[start:end]
        if len(part) <= max_size:
            spans.append((start, end))
            continue
        for s, e in pack(part, blank_line_cuts(part), max_size):
            spans.append((start + s, start + e))
    return spans


def chunk_file(file_path: str, text: str, max_size: int = 2000) -> List[Chunk]:
    """Split one file into chunks, picking the strategy by extension."""
    if file_path.endswith(".py"):
        spans = pack(text, python_cuts(text), max_size)
    else:
        spans = markdown_spans(text, max_size)
    chunks: List[Chunk] = []
    for first, last in spans:
        piece = text[first:last]
        if piece.strip():
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=first,
                    last_character_index=last,
                    text=piece,
                )
            )
    return chunks
