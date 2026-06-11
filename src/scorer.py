"""
scorer.py — Combine FAISS semantic similarity with feature-engineered scores.

Two-stage pipeline:
  Stage 1 (FAISS): Retrieve top-K semantically similar candidates.
                   Fast cosine similarity in embedding space.
  Stage 2 (Features): Re-rank the top-K using interpretable feature scores.

Why two stages:
  - FAISS catches candidates whose career descriptions semantically match
    the JD even if they don't use exact keywords.
  - Feature scoring catches the signals FAISS can't see:
    behavioral signals, notice period, location, company type.

Final score = alpha * semantic_score + (1 - alpha) * feature_score
where alpha = 0.40 (semantic) + 0.60 (features).

We weight features higher because:
  1. Behavioral signals (availability) are not captured by text
  2. The JD explicitly warns against keyword-based matching
  3. Company type (consulting vs product) needs explicit rules
"""

from typing import Dict, Any, List, Tuple
import numpy as np


# Blend weight: how much to trust semantic vs feature scores
SEMANTIC_WEIGHT = 0.40
FEATURE_WEIGHT = 0.60

# Number of candidates to retrieve from FAISS before re-ranking
FAISS_TOP_K = 500


def blend_scores(
    semantic_score: float,
    feature_score: float
) -> float:
    """
    Blend semantic similarity with feature-engineered score.
    
    Args:
        semantic_score: Cosine similarity from FAISS (0.0-1.0)
        feature_score:  Weighted feature score from features.py (0.0-1.0)
    
    Returns:
        Blended final score (0.0-1.0)
    """
    return SEMANTIC_WEIGHT * semantic_score + FEATURE_WEIGHT * feature_score


def rank_candidates(
    scored_candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Sort candidates by final_score descending, assign ranks 1-100.
    
    Args:
        scored_candidates: List of dicts with 'candidate_id', 'final_score',
                           'feature_vector', 'reasoning'
    
    Returns:
        Top 100 candidates sorted by rank, with rank field added.
    """
    # Sort by final_score descending
    sorted_candidates = sorted(
        scored_candidates,
        key=lambda x: x["final_score"],
        reverse=True
    )

    # Take top 100
    top_100 = sorted_candidates[:100]

    # Assign ranks 1-100
    for i, candidate in enumerate(top_100):
        candidate["rank"] = i + 1

    # Tie-break: if two candidates share the same score, sort by candidate_id
    # (ascending) per submission_spec.md
    _resolve_ties(top_100)

    return top_100


def _resolve_ties(ranked: List[Dict[str, Any]]) -> None:
    """
    In-place tie resolution: equal scores → sort by candidate_id ascending.
    Required by submission_spec.md.
    """
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and abs(ranked[j]["final_score"] - ranked[i]["final_score"]) < 1e-9:
            j += 1

        # Sort the tied group by candidate_id ascending
        if j - i > 1:
            tied_group = ranked[i:j]
            tied_group.sort(key=lambda x: x["candidate_id"])
            ranked[i:j] = tied_group
            # Re-assign ranks within tied group
            for k in range(i, j):
                ranked[k]["rank"] = k + 1

        i = j


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Min-max normalize a list of scores to [0.0, 1.0].
    Ensures the final score column is monotonically non-increasing with rank.
    """
    if not scores:
        return scores
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def ensure_monotonic(ranked: List[Dict[str, Any]]) -> None:
    """
    Ensure scores are monotonically non-increasing with rank.
    Required by submission_spec.md: score at rank 1 >= score at rank 2 >= ...
    
    Applied in-place by clipping each score to be at most the previous score.
    """
    if not ranked:
        return
    prev_score = ranked[0]["final_score"]
    for candidate in ranked[1:]:
        if candidate["final_score"] > prev_score:
            candidate["final_score"] = prev_score
        else:
            prev_score = candidate["final_score"]
