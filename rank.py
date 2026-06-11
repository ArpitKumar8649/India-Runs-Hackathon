#!/usr/bin/env python3
"""
rank.py — Main entry point for the Redrob Hackathon ranker.

Usage:
    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv --team-id team_yourname

What this script does:
    1. Loads all 100,000 candidates from the JSONL file
    2. Loads pre-built FAISS index from artifacts/ (built in Colab)
    3. Embeds the JD query text using SentenceTransformer
    4. Stage 1: FAISS retrieves top-500 semantically similar candidates
    5. Stage 2: Feature scoring on all 500 candidates
    6. Honeypot detection and penalization
    7. Blend semantic + feature scores → final ranking
    8. Generate per-candidate reasoning strings
    9. Write top-100 to submission CSV
    10. Validate format using submission_spec rules

Compute constraints (must satisfy):
    - CPU only (no GPU)
    - <= 5 minutes wall-clock
    - <= 16 GB RAM
    - No network calls during ranking

Runtime estimate on a 16GB CPU machine:
    - Loading 100k candidates: ~30 seconds
    - FAISS search: <1 second
    - Feature scoring (500 candidates): ~5 seconds
    - Total: well under 5 minutes
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.loader import load_candidates, build_id_index
from src.text_builder import build_jd_query_text
from src.honeypot_detector import detect_honeypot, score_honeypot_risk
from src.features import compute_features
from src.scorer import blend_scores, rank_candidates, ensure_monotonic
from src.reasoner import generate_reasoning


# ── Constants ─────────────────────────────────────────────────────────────────

ARTIFACTS_DIR = Path("artifacts")
FAISS_INDEX_PATH = ARTIFACTS_DIR / "candidates.faiss"
CANDIDATE_IDS_PATH = ARTIFACTS_DIR / "candidate_ids.json"
MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_TOP_K = 500  # Retrieve top-500, then re-rank to top-100


def parse_args():
    parser = argparse.ArgumentParser(
        description="Redrob Hackathon — Candidate Ranker"
    )
    parser.add_argument(
        "--candidates",
        default="./candidates.jsonl.gz",
        help="Path to candidates.jsonl or candidates.jsonl.gz"
    )
    parser.add_argument(
        "--out",
        default="./submission.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--team-id",
        default="team_submission",
        help="Your registered participant ID (used as filename)"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run on sample_candidates.json only (for testing)"
    )
    return parser.parse_args()


def load_faiss_artifacts():
    """Load pre-built FAISS index and ordered candidate IDs from artifacts/."""
    if not FAISS_INDEX_PATH.exists():
        print(
            f"\n[ERROR] FAISS index not found at {FAISS_INDEX_PATH}\n"
            f"You need to run the pre-computation step first.\n"
            f"Open notebooks/01_precompute_embeddings.ipynb in Google Colab,\n"
            f"run all cells, then download candidates.faiss and candidate_ids.json\n"
            f"and place them in the artifacts/ folder.\n"
        )
        sys.exit(1)

    if not CANDIDATE_IDS_PATH.exists():
        print(f"[ERROR] candidate_ids.json not found at {CANDIDATE_IDS_PATH}")
        sys.exit(1)

    print("Loading FAISS index...")
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    with open(CANDIDATE_IDS_PATH, "r") as f:
        candidate_ids_ordered = json.load(f)

    print(f"  Index has {index.ntotal:,} vectors")
    print(f"  Candidate IDs loaded: {len(candidate_ids_ordered):,}")
    return index, candidate_ids_ordered


def embed_jd_query(model: SentenceTransformer) -> np.ndarray:
    """Embed the JD query text for FAISS search."""
    query_text = build_jd_query_text()
    print(f"Embedding JD query ({len(query_text)} chars)...")
    embedding = model.encode([query_text], normalize_embeddings=True)
    return embedding.astype("float32")


def stage1_faiss_retrieval(
    index: faiss.Index,
    candidate_ids_ordered: list,
    query_embedding: np.ndarray,
    top_k: int
) -> list:
    """
    Stage 1: Fast FAISS semantic search.
    Returns list of (candidate_id, semantic_score) tuples.
    """
    print(f"Stage 1: FAISS retrieval (top {top_k})...")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(candidate_ids_ordered):
            cid = candidate_ids_ordered[idx]
            results.append((cid, float(dist)))

    print(f"  Retrieved {len(results)} candidates from FAISS")
    return results


def stage2_feature_ranking(
    faiss_results: list,
    id_index: dict
) -> list:
    """
    Stage 2: Feature-based re-ranking of FAISS top-K.
    
    For each candidate:
    1. Detect honeypot risk
    2. Compute feature vector (skill, career, availability, location)
    3. Blend semantic + feature scores
    4. Generate reasoning
    
    Returns list of scored candidate dicts.
    """
    print(f"Stage 2: Feature scoring ({len(faiss_results)} candidates)...")

    scored = []
    honeypot_count = 0

    for cid, semantic_score in faiss_results:
        candidate = id_index.get(cid)
        if not candidate:
            continue

        # Honeypot detection
        honeypot_risk = score_honeypot_risk(candidate)
        is_hp, hp_reasons = detect_honeypot(candidate)
        if is_hp:
            honeypot_count += 1

        # Feature scoring
        fv = compute_features(candidate)
        fv.honeypot_risk = honeypot_risk

        # Recompute final score with honeypot penalty applied
        from src.features import _combine_scores
        fv.final_score = _combine_scores(fv)

        # Blend semantic + feature
        blended = blend_scores(semantic_score, fv.final_score)

        scored.append({
            "candidate_id": cid,
            "semantic_score": semantic_score,
            "feature_score": fv.final_score,
            "final_score": blended,
            "feature_vector": fv,
            "candidate": candidate,
            "is_honeypot": is_hp,
            "honeypot_reasons": hp_reasons
        })

    print(f"  Honeypots detected in top-{FAISS_TOP_K}: {honeypot_count}")
    return scored


def build_submission_csv(
    ranked: list,
    out_path: str,
    team_id: str
) -> None:
    """
    Write the final submission CSV.
    
    Format: candidate_id, rank, score, reasoning
    All scores monotonically non-increasing.
    Exactly 100 rows.
    """
    # Ensure scores are monotonically non-increasing
    ensure_monotonic(ranked)

    # Generate reasoning for each candidate
    rows = []
    for entry in ranked:
        candidate = entry["candidate"]
        fv = entry["feature_vector"]
        rank = entry["rank"]
        score = entry["final_score"]

        reasoning = generate_reasoning(candidate, fv, rank, score)

        rows.append({
            "candidate_id": entry["candidate_id"],
            "rank": rank,
            "score": round(score, 6),
            "reasoning": reasoning
        })

    # Write CSV
    out_path_obj = Path(out_path)

    # If team_id provided, use that as filename
    if team_id != "team_submission":
        out_path_obj = out_path_obj.parent / f"{team_id}.csv"

    with open(out_path_obj, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Submission written to: {out_path_obj}")
    print(f"   Rows: {len(rows)}")
    print(f"   Score range: {rows[-1]['score']:.4f} — {rows[0]['score']:.4f}")


def validate_output(out_path: str) -> bool:
    """
    Quick in-process validation of the output CSV.
    Mirrors the logic in validate_submission.py.
    """
    import csv
    import re

    errors = []
    cid_pattern = re.compile(r"^CAND_[0-9]{7}$")
    required_header = ["candidate_id", "rank", "score", "reasoning"]

    with open(out_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != required_header:
            errors.append(f"Header mismatch: {header}")
        rows = [r for r in reader if any(c.strip() for c in r)]

    if len(rows) != 100:
        errors.append(f"Expected 100 rows, got {len(rows)}")

    ranks_seen = set()
    ids_seen = set()
    for i, row in enumerate(rows):
        cid, rank_s, score_s, reasoning = row[0], row[1], row[2], row[3]
        if not cid_pattern.match(cid):
            errors.append(f"Row {i+2}: bad candidate_id: {cid}")
        if cid in ids_seen:
            errors.append(f"Row {i+2}: duplicate candidate_id: {cid}")
        ids_seen.add(cid)
        rank = int(rank_s)
        if rank in ranks_seen:
            errors.append(f"Row {i+2}: duplicate rank: {rank}")
        ranks_seen.add(rank)

    if errors:
        print("\n⚠️  Validation warnings:")
        for e in errors:
            print(f"   - {e}")
        return False

    print("✅ Submission format is valid.")
    return True


def print_top_10(ranked: list) -> None:
    """Pretty-print the top 10 candidates for review."""
    print("\n" + "=" * 80)
    print("TOP 10 CANDIDATES")
    print("=" * 80)
    for entry in ranked[:10]:
        c = entry["candidate"]
        p = c["profile"]
        fv = entry["feature_vector"]
        print(
            f"#{entry['rank']:2d} [{entry['final_score']:.4f}] "
            f"{p['current_title'][:30]:<30} | "
            f"{p['years_of_experience']:.1f}yr | "
            f"{p['location'][:20]:<20} | "
            f"Skills:{fv.skill_match_score:.2f} "
            f"Career:{fv.career_fit_score:.2f} "
            f"Avail:{fv.availability_score:.2f}"
        )
        print(f"   💬 {entry['reasoning'][:90]}")
    print("=" * 80)


def main():
    start_time = time.time()
    args = parse_args()

    print("\n" + "=" * 60)
    print("  REDROB HACKATHON — CANDIDATE RANKER")
    print("=" * 60)

    # ── Step 1: Load candidates ────────────────────────────────────────────
    if args.sample:
        from src.loader import load_sample
        candidates = load_sample("sample_candidates.json")
    else:
        candidates = load_candidates(args.candidates)

    id_index = build_id_index(candidates)

    # ── Step 2: Load FAISS index ───────────────────────────────────────────
    faiss_index, candidate_ids_ordered = load_faiss_artifacts()

    # ── Step 3: Embed JD query ─────────────────────────────────────────────
    print(f"Loading model: {MODEL_NAME} (CPU)...")
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = embed_jd_query(model)

    # ── Step 4: Stage 1 — FAISS retrieval ─────────────────────────────────
    faiss_results = stage1_faiss_retrieval(
        faiss_index, candidate_ids_ordered, query_embedding, FAISS_TOP_K
    )

    # ── Step 5: Stage 2 — Feature re-ranking ──────────────────────────────
    scored = stage2_feature_ranking(faiss_results, id_index)

    # ── Step 6: Rank and generate reasoning ───────────────────────────────
    print("Ranking and generating reasoning...")
    ranked = rank_candidates(scored)

    # Add reasoning to each ranked entry
    for entry in ranked:
        entry["reasoning"] = generate_reasoning(
            entry["candidate"],
            entry["feature_vector"],
            entry["rank"],
            entry["final_score"]
        )

    # ── Step 7: Print top 10 for review ───────────────────────────────────
    print_top_10(ranked)

    # ── Step 8: Write submission CSV ──────────────────────────────────────
    build_submission_csv(ranked, args.out, args.team_id)

    # ── Step 9: Validate output ───────────────────────────────────────────
    output_file = args.out
    if args.team_id != "team_submission":
        output_file = str(Path(args.out).parent / f"{args.team_id}.csv")
    validate_output(output_file)

    # ── Step 10: Print runtime ────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n⏱  Total runtime: {elapsed:.1f}s  (limit: 300s)")
    if elapsed > 240:
        print("⚠️  WARNING: Approaching 5-minute limit. Optimize if needed.")
    else:
        print(f"✅ Well within compute budget ({300 - elapsed:.0f}s remaining)")


if __name__ == "__main__":
    main()
