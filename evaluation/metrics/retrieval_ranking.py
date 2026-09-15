"""Pure retrieval ranking metrics for RAG evaluation."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Any

from nagrik_ai.utils.source_types import classify_source_type

AUTHORITATIVE_SOURCE_TYPES: frozenset[str] = frozenset({"act", "rules", "notification"})
DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)
_AUTHORITY_GRADES: dict[str, float] = {
    "act": 3.0,
    "rules": 2.0,
    "notification": 1.0,
}


def _normalize_id(raw: str | int) -> str:
    """Normalize IDs for exact source-id matching."""
    return str(raw).strip().lower()


def _doc_source_id(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata", {})
    if not isinstance(metadata, dict):
        return ""
    raw = metadata.get("source_id", metadata.get("source", ""))
    if not isinstance(raw, (str, int)):
        return ""
    return _normalize_id(raw)


def _normalize_relevant_ids(relevant_ids: set[str] | set[int]) -> set[str]:
    return {_normalize_id(raw) for raw in relevant_ids if _normalize_id(raw)}


def _normalize_graded_map(graded_map: dict[str, float] | None) -> dict[str, float]:
    if graded_map is None:
        return {}
    return {_normalize_id(key): float(value) for key, value in graded_map.items() if _normalize_id(key)}


def _rel_vector(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    graded: dict[str, float] | None = None,
) -> list[float]:
    grades = _normalize_graded_map(graded)
    return [grades.get(doc_id, 1.0 if doc_id in relevant_ids else 0.0) for doc_id in retrieved_ids]


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Precision@k with k clamped to the retrieved list length."""
    if k <= 0 or not retrieved_ids:
        return 0.0
    normalized_relevant = _normalize_relevant_ids(relevant_ids)
    top = [_normalize_id(doc_id) for doc_id in retrieved_ids[: min(k, len(retrieved_ids))]]
    if not top:
        return 0.0
    hits = sum(1 for doc_id in top if doc_id in normalized_relevant)
    return hits / len(top)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@k over the known relevant ID set."""
    normalized_relevant = _normalize_relevant_ids(relevant_ids)
    if k <= 0 or not normalized_relevant:
        return 0.0
    top = {_normalize_id(doc_id) for doc_id in retrieved_ids[: min(k, len(retrieved_ids))]}
    hits = len(normalized_relevant & top)
    return hits / len(normalized_relevant)


def average_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Average precision for one query; returns 0.0 when no relevant IDs exist."""
    normalized_relevant = _normalize_relevant_ids(relevant_ids)
    if not normalized_relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate((_normalize_id(raw) for raw in retrieved_ids), start=1):
        if doc_id in normalized_relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(normalized_relevant)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant retrieved ID."""
    normalized_relevant = _normalize_relevant_ids(relevant_ids)
    if not normalized_relevant:
        return 0.0
    for rank, doc_id in enumerate((_normalize_id(raw) for raw in retrieved_ids), start=1):
        if doc_id in normalized_relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(rels: list[float], k: int, graded: bool = False) -> float:
    """Discounted cumulative gain at k.

    Binary relevance uses linear gains. Graded relevance uses exponential gains.
    """
    if k <= 0:
        return 0.0
    total = 0.0
    for index, rel in enumerate(rels[:k], start=1):
        gain = (2.0**rel) - 1.0 if graded else rel
        total += gain / math.log2(index + 1)
    return total


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
    graded_map: dict[str, float] | None = None,
) -> float:
    """Normalized DCG at k, binary unless a graded map is supplied."""
    normalized_relevant = _normalize_relevant_ids(relevant_ids)
    if k <= 0 or not normalized_relevant:
        return 0.0
    retrieved = [_normalize_id(doc_id) for doc_id in retrieved_ids]
    grades = _normalize_graded_map(graded_map)
    graded = bool(grades)
    rels = _rel_vector(retrieved, normalized_relevant, grades if graded else None)
    if graded:
        ideal_rels = sorted((grades.get(doc_id, 1.0) for doc_id in normalized_relevant), reverse=True)
    else:
        ideal_rels = [1.0] * len(normalized_relevant)
    ideal = dcg_at_k(ideal_rels, k, graded=graded)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(rels, k, graded=graded) / ideal


def default_authority_grade(authority: str) -> float:
    """Default nDCG grade for an authority/source-type label."""
    return _AUTHORITY_GRADES.get(authority.strip().lower().replace(" ", "_"), 0.0)


def retrieval_ranking_metrics(
    retrieved: list[dict[str, Any]],
    relevant_ids: set[str] | set[int],
    ks: tuple[int, ...] = DEFAULT_KS,
    graded_map: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute ranking metrics for one retrieved result list."""
    retrieved_ids = [_doc_source_id(doc) for doc in retrieved]
    retrieved_ids = [doc_id for doc_id in retrieved_ids if doc_id]
    normalized_relevant = _normalize_relevant_ids(relevant_ids)
    metrics: dict[str, float] = {
        "ap": average_precision(retrieved_ids, normalized_relevant),
        "rr": reciprocal_rank(retrieved_ids, normalized_relevant),
        "num_relevant": float(len(normalized_relevant)),
        "num_retrieved": float(len(retrieved_ids)),
    }
    for k in ks:
        metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, normalized_relevant, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, normalized_relevant, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, normalized_relevant, k, graded_map=graded_map)
        metrics[f"hit_rate@{k}"] = 1.0 if metrics[f"recall@{k}"] > 0.0 else 0.0
    return metrics


def aggregate_ranking_metrics(per_case: list[dict[str, float]]) -> dict[str, float]:
    """Mean retrieval ranking metrics across cases.

    MAP excludes cases with no relevant IDs. Other means include all supplied cases.
    """
    if not per_case:
        return {}
    metric_names = sorted({name for metrics in per_case for name in metrics})
    aggregate: dict[str, float] = {}
    for name in metric_names:
        if name == "ap":
            values = [
                metrics[name] for metrics in per_case if metrics.get("num_relevant", 0.0) > 0.0 and name in metrics
            ]
            aggregate["map"] = fmean(values) if values else 0.0
            continue
        output_name = "mrr" if name == "rr" else name
        values = [metrics[name] for metrics in per_case if name in metrics]
        if values:
            aggregate[output_name] = fmean(values)
    return aggregate


def is_authoritative_doc(doc: dict[str, Any]) -> bool:
    """True when a retrieved doc is an Act, Rules, or Notification chunk."""
    return classify_source_type(doc.get("metadata", {})) in AUTHORITATIVE_SOURCE_TYPES


def authority_atk(docs: list[dict[str, Any]], k: int | None = None) -> float:
    """Authority@k: proportion of top-k chunks from Acts/Rules/Notifications."""
    if not docs:
        return 0.0
    top = docs[:k] if k is not None else docs
    if not top:
        return 0.0
    return sum(1 for doc in top if is_authoritative_doc(doc)) / len(top)


def first_authoritative_rank(docs: list[dict[str, Any]]) -> int | None:
    """Position (1-based) of the first authoritative source; None when absent."""
    for rank, doc in enumerate(docs, start=1):
        if is_authoritative_doc(doc):
            return rank
    return None


def retrieval_metrics(docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic authority-proxy metrics over the final result list."""
    authoritative_count = sum(1 for doc in docs if is_authoritative_doc(doc))
    return {
        "authority_at_top_k": authority_atk(docs),
        "authority_at_1": authority_atk(docs, k=1),
        "authority_at_3": authority_atk(docs, k=3),
        "authority_at_5": authority_atk(docs, k=5),
        "first_authoritative_rank": first_authoritative_rank(docs),
        "num_authoritative": authoritative_count,
        "total": len(docs),
    }
