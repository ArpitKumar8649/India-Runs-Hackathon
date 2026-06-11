"""
app.py — HuggingFace Spaces Gradio demo for the Redrob Hackathon.

This is the required "sandbox link" for your submission.
It accepts a small candidate sample (≤100 candidates) and runs
the full ranking pipeline, outputting a ranked CSV.

Deploy to HuggingFace Spaces:
  1. Create a new Space: https://huggingface.co/new-space
  2. Select Gradio SDK
  3. Upload: app.py, requirements.txt, artifacts/ folder
  4. The Space will auto-build and run

The full 100k candidate ranking is NOT done here — the sandbox
only needs to demonstrate the pipeline works on a small sample.
"""

import gradio as gr
import json
import csv
import io
import sys
import os
from pathlib import Path
from datetime import date

# ── Import our pipeline modules ─────────────────────────────────────────────
# (These modules are in src/ — same codebase as rank.py)
sys.path.insert(0, str(Path(__file__).parent))

from src.text_builder import build_jd_query_text
from src.honeypot_detector import detect_honeypot, score_honeypot_risk
from src.features import compute_features, _combine_scores
from src.scorer import blend_scores, rank_candidates, ensure_monotonic
from src.reasoner import generate_reasoning

# ── Load model once at startup ───────────────────────────────────────────────
print("Loading SentenceTransformer model...")
from sentence_transformers import SentenceTransformer
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

JD_TEXT = build_jd_query_text()
JD_EMBEDDING = MODEL.encode([JD_TEXT], normalize_embeddings=True)


def rank_sample(candidates_json_text: str) -> tuple:
    """
    Core ranking function used by the Gradio interface.
    
    Args:
        candidates_json_text: JSON string of a list of candidate dicts
    
    Returns:
        (ranked_csv_string, status_message, top10_table)
    """
    try:
        # Parse input
        candidates = json.loads(candidates_json_text)
        if not isinstance(candidates, list):
            return "", "❌ Input must be a JSON array of candidates.", ""

        if len(candidates) > 100:
            candidates = candidates[:100]

        if len(candidates) == 0:
            return "", "❌ No candidates provided.", ""

        # Build texts and embeddings for this sample
        from src.text_builder import build_candidate_text
        texts = [build_candidate_text(c) for c in candidates]
        embeddings = MODEL.encode(texts, normalize_embeddings=True)

        # Compute semantic similarities
        import numpy as np
        similarities = (embeddings @ JD_EMBEDDING.T).squeeze().tolist()
        if isinstance(similarities, float):
            similarities = [similarities]

        # Score each candidate
        scored = []
        for candidate, semantic_score in zip(candidates, similarities):
            honeypot_risk = score_honeypot_risk(candidate)
            fv = compute_features(candidate)
            fv.honeypot_risk = honeypot_risk
            fv.final_score = _combine_scores(fv)
            blended = blend_scores(float(semantic_score), fv.final_score)

            scored.append({
                "candidate_id": candidate["candidate_id"],
                "semantic_score": float(semantic_score),
                "feature_score": fv.final_score,
                "final_score": blended,
                "feature_vector": fv,
                "candidate": candidate,
                "is_honeypot": honeypot_risk > 0.3,
                "honeypot_reasons": []
            })

        # Rank
        ranked = rank_candidates(scored)
        ensure_monotonic(ranked)

        # Add reasoning
        for entry in ranked:
            entry["reasoning"] = generate_reasoning(
                entry["candidate"],
                entry["feature_vector"],
                entry["rank"],
                entry["final_score"]
            )

        # Build CSV output
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for entry in ranked:
            writer.writerow([
                entry["candidate_id"],
                entry["rank"],
                round(entry["final_score"], 6),
                entry["reasoning"]
            ])
        csv_string = output.getvalue()

        # Build top-10 table for display
        table_rows = []
        for entry in ranked[:10]:
            p = entry["candidate"]["profile"]
            fv = entry["feature_vector"]
            hp_flag = " 🚨" if entry.get("is_honeypot") else ""
            table_rows.append([
                entry["rank"],
                entry["candidate_id"],
                p["current_title"],
                f"{p['years_of_experience']:.1f}",
                p["location"],
                f"{entry['final_score']:.4f}",
                f"S:{fv.skill_match_score:.2f} C:{fv.career_fit_score:.2f} A:{fv.availability_score:.2f}" + hp_flag,
                entry["reasoning"][:80] + "..."
            ])

        status = (
            f"✅ Ranked {len(ranked)} candidates | "
            f"Honeypots detected: {sum(1 for e in ranked if e.get('is_honeypot'))} | "
            f"Time: complete"
        )

        return csv_string, status, table_rows

    except json.JSONDecodeError as e:
        return "", f"❌ JSON parse error: {e}", []
    except Exception as e:
        import traceback
        return "", f"❌ Error: {e}\n{traceback.format_exc()}", []


def load_sample_file():
    """Load the bundled sample_candidates.json for demonstration."""
    sample_path = "sample_candidates.json"
    if os.path.exists(sample_path):
        with open(sample_path, "r") as f:
            return f.read()
    return "[]"


# ── Gradio Interface ──────────────────────────────────────────────────────────

with gr.Blocks(title="Redrob Hackathon — Candidate Ranker") as demo:

    gr.Markdown("""
    # 🔍 Redrob Intelligent Candidate Ranker
    ### Hackathon submission sandbox — India Runs × Hack2Skill

    **Pipeline:** Two-stage ranking using semantic embeddings (all-MiniLM-L6-v2 + FAISS)
    + feature-based re-ranking (23 Redrob behavioral signals + JD-derived career scoring).

    **This sandbox accepts up to 100 candidates** and produces a ranked CSV matching
    the submission spec format.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            candidates_input = gr.Textbox(
                label="Candidates JSON (paste sample_candidates.json content)",
                placeholder='[{"candidate_id": "CAND_0000001", ...}, ...]',
                lines=15
            )

            with gr.Row():
                load_btn = gr.Button("📂 Load Sample Data", variant="secondary")
                rank_btn = gr.Button("🚀 Rank Candidates", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Output")
            status_output = gr.Textbox(label="Status", lines=2)
            csv_output = gr.Textbox(
                label="Ranked CSV Output (copy to save)",
                lines=12
            )

    gr.Markdown("### Top 10 Candidates")
    top10_table = gr.Dataframe(
        headers=["Rank", "ID", "Title", "YoE", "Location",
                 "Score", "Component Scores", "Reasoning"],
        label="Top 10 Ranked Candidates"
    )

    gr.Markdown("""
    ---
    **Scoring breakdown:**
    - **S:** Skill match score (required + bonus AI/retrieval skills)
    - **C:** Career fit score (product company, retrieval work, YoE 5-9 range)
    - **A:** Availability score (last active date, open to work, notice period, response rate)
    - 🚨 Honeypot flag (impossible profile detected — penalized)
    """)

    # Event handlers
    load_btn.click(fn=load_sample_file, outputs=candidates_input)
    rank_btn.click(
        fn=rank_sample,
        inputs=candidates_input,
        outputs=[csv_output, status_output, top10_table]
    )

if __name__ == "__main__":
    demo.launch()
