"""
loader.py — Load and parse candidate profiles from candidates.jsonl.gz

Handles both:
- candidates.jsonl.gz  (compressed, ~52 MB)
- candidates.jsonl     (uncompressed, ~465 MB)

Returns a list of dicts with the full candidate schema.
"""

import gzip
import json
import os
from typing import List, Dict, Any
from tqdm import tqdm


def load_candidates(path: str, limit: int = None) -> List[Dict[str, Any]]:
    """
    Load all candidates from a .jsonl or .jsonl.gz file.

    Args:
        path:  Path to candidates.jsonl or candidates.jsonl.gz
        limit: If set, only load the first N candidates (useful for testing)

    Returns:
        List of candidate dicts matching the Redrob candidate schema.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find candidates file at: {path}\n"
            f"Make sure candidates.jsonl.gz is in your project root."
        )

    candidates = []
    open_fn = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"

    print(f"Loading candidates from: {path}")

    with open_fn(path, mode, encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Loading", unit=" candidates")):
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
                candidates.append(candidate)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line {i+1}: {e}")
                continue

            if limit and len(candidates) >= limit:
                break

    print(f"Loaded {len(candidates):,} candidates.")
    return candidates


def load_sample(sample_path: str) -> List[Dict[str, Any]]:
    """
    Load sample_candidates.json (pretty-printed JSON array, not JSONL).
    Useful for rapid development and testing without the full 465 MB file.
    """
    with open(sample_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    print(f"Loaded {len(candidates)} sample candidates.")
    return candidates


def get_candidate_ids(candidates: List[Dict[str, Any]]) -> List[str]:
    """Extract ordered list of candidate_ids."""
    return [c["candidate_id"] for c in candidates]


def build_id_index(candidates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build a dict of candidate_id -> candidate for O(1) lookup.
    Used during scoring to fetch full profiles by ID.
    """
    return {c["candidate_id"]: c for c in candidates}
