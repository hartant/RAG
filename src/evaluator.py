"""Compute recall@k of search results against ground truth."""

from typing import Dict, List

from .models import AnsweredQuestion, MinimalSource, StudentSearchResults

IOU_THRESHOLD = 0.05


def iou(a: MinimalSource, b: MinimalSource) -> float:
    """Intersection over union of two ranges in the same file."""
    inter = min(a.last_character_index, b.last_character_index) - max(
        a.first_character_index, b.first_character_index
    )
    if inter <= 0:
        return 0.0
    union = (
        (a.last_character_index - a.first_character_index)
        + (b.last_character_index - b.first_character_index)
        - inter
    )
    return inter / union if union > 0 else 0.0


def is_found(truth: MinimalSource, results: List[MinimalSource]) -> bool:
    """Return True if a result is in the same file and overlaps truth."""
    for r in results:
        if r.file_path == truth.file_path and iou(truth, r) >= IOU_THRESHOLD:
            return True
    return False


def recall_at_k(
    student: StudentSearchResults,
    truth: Dict[str, AnsweredQuestion],
    k: int,
) -> float:
    """Average recall@k over every question that has ground truth."""
    scores: List[float] = []
    for res in student.search_results:
        question = truth.get(res.question_id)
        if question is None or not question.sources:
            continue
        top = res.retrieved_sources[:k]
        found = sum(1 for s in question.sources if is_found(s, top))
        scores.append(found / len(question.sources))
    return sum(scores) / len(scores) if scores else 0.0
