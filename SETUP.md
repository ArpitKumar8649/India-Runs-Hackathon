# Redrob Ranker — Full Setup Guide

## Project Structure

```
redrob_ranker/
├── SETUP.md                          ← You are here
├── rank.py                           ← MAIN: python rank.py → submission.csv
├── requirements.txt                  ← All dependencies
├── submission_metadata.yaml          ← Fill this before submitting
├── validate_submission.py            ← Provided by Redrob (copy here)
│
├── src/
│   ├── loader.py                     ← Load + parse candidates.jsonl.gz
│   ├── text_builder.py               ← Build rich text from candidate fields
│   ├── honeypot_detector.py          ← Detect impossible/fake profiles
│   ├── features.py                   ← Score all 23 signals + career fit
│   ├── scorer.py                     ← Combine features into final score
│   └── reasoner.py                   ← Generate per-candidate reasoning text
│
├── artifacts/                        ← Pre-computed files (from Colab)
│   ├── candidates.faiss              ← FAISS index (generated in Colab)
│   └── candidate_ids.json            ← Ordered candidate IDs matching index
│
├── notebooks/
│   └── 01_precompute_embeddings.ipynb ← Run ONCE in Google Colab
│
└── app.py                            ← HuggingFace Spaces Gradio demo
```

---

## Phase 0 — Local VS Code Environment Setup

### Step 1: Install Python 3.10+
```
https://www.python.org/downloads/
```
Make sure to check "Add Python to PATH" during installation.

### Step 2: Create project folder
```bash
mkdir redrob_ranker
cd redrob_ranker
```

### Step 3: Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install all dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Place your data files
```
redrob_ranker/
└── candidates.jsonl.gz     ← Put the downloaded dataset HERE
```

### Step 6: Copy validate_submission.py
Copy the `validate_submission.py` from the hackathon bundle to your project root.

---

## Phase 1 — Pre-computation (Google Colab, Run ONCE)

Open `notebooks/01_precompute_embeddings.ipynb` in Google Colab.
This generates `candidates.faiss` and `candidate_ids.json`.
Download both files and place them in `artifacts/`.

---

## Phase 2 — Local Ranking (VS Code, <5 min on CPU)

```bash
python rank.py --candidates ./candidates.jsonl.gz --out ./my_submission.csv
python validate_submission.py my_submission.csv
```

---

## Phase 3 — Sandbox Demo (HuggingFace Spaces)

Push `app.py`, `requirements.txt`, and `artifacts/` to a HuggingFace Space.
The space uses pre-built artifacts — no GPU needed.
