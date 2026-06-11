# Redrob Intelligent Candidate Ranker
### India Runs × Hack2Skill — Data & AI Challenge

A two-stage semantic + feature ranking system that finds the best
Senior AI Engineer candidates from a 100k-candidate pool using
embedding-based retrieval and JD-aligned feature scoring.

---

## Architecture

```
candidates.jsonl.gz (100k)
         │
         ▼  [Pre-computation — Google Colab, run once]
  ┌─────────────────────────────────────────────┐
  │  Build rich candidate text                   │
  │  (title + summary + career + weighted skills)│
  │          ↓                                   │
  │  Embed with all-MiniLM-L6-v2               │
  │  (384-dim vectors, normalized)               │
  │          ↓                                   │
  │  Build FAISS IndexFlatIP                     │
  │  Save candidates.faiss + candidate_ids.json  │
  └─────────────────────────────────────────────┘
         │
         ▼  [Ranking step — local CPU, < 5 minutes]
  ┌─────────────────────────────────────────────┐
  │  Embed JD query text                         │
  │          ↓                                   │
  │  STAGE 1: FAISS → top-500 candidates         │
  │  (fast cosine similarity)                    │
  │          ↓                                   │
  │  Honeypot detection                          │
  │  (impossible tenure, skill inflation, etc.)  │
  │          ↓                                   │
  │  STAGE 2: Feature scoring                    │
  │  • Skill match     (35%)                     │
  │  • Career fit      (35%)                     │
  │  • Availability    (20%)                     │
  │  • Location        (10%)                     │
  │          ↓                                   │
  │  Blend: 40% semantic + 60% feature           │
  │          ↓                                   │
  │  Generate per-candidate reasoning            │
  │          ↓                                   │
  │  Output top-100 ranked CSV                   │
  └─────────────────────────────────────────────┘
```

---

## Reproduce the Submission

### Requirements
- Python 3.10+
- 16 GB RAM (CPU only — no GPU required for ranking)

### Setup
```bash
git clone https://github.com/YOUR_USERNAME/redrob-ranker
cd redrob-ranker

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Pre-computation (Google Colab — run once)
1. Upload `candidates.jsonl.gz` to Google Drive
2. Open `notebooks/01_precompute_embeddings.ipynb` in Colab
3. Set Runtime → T4 GPU
4. Update `CANDIDATES_PATH` and `OUTPUT_DIR` in Cell 2
5. Run all cells (~10 minutes)
6. Download `candidates.faiss` and `candidate_ids.json`
7. Place both files in `artifacts/`

### Run the Ranker
```bash
python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
```

### Validate
```bash
python validate_submission.py submission.csv
```

---

## Project Structure

```
redrob_ranker/
├── rank.py                        ← Main entry point → produces CSV
├── app.py                         ← HuggingFace Spaces sandbox demo
├── requirements.txt
├── submission_metadata.yaml       ← Fill before submitting
├── validate_submission.py         ← Provided by Redrob
│
├── src/
│   ├── loader.py                  ← Load candidates.jsonl.gz
│   ├── text_builder.py            ← Rich text for embedding
│   ├── honeypot_detector.py       ← Detect impossible profiles
│   ├── features.py                ← Feature engineering + scoring
│   ├── scorer.py                  ← Blend scores + rank
│   └── reasoner.py                ← Generate reasoning strings
│
├── artifacts/
│   ├── candidates.faiss           ← Pre-built FAISS index
│   └── candidate_ids.json         ← Ordered candidate IDs
│
└── notebooks/
    └── 01_precompute_embeddings.ipynb  ← Run in Colab
```

---

## Scoring Design

### Why 35% Skill + 35% Career + 20% Availability + 10% Location?

The weights directly reflect the JD's stated priorities:

**Skill Match (35%):**
- Required skills (FAISS, Pinecone, sentence-transformers, etc.) carry full weight
- Proficiency level matters: expert > advanced > intermediate > beginner
- Duration months validates claimed proficiency
- Redrob platform assessment scores used as verification signal

**Career Fit (35%):**
- The JD explicitly disqualifies consulting-only careers (TCS/Infosys/Wipro etc.) → 85% score penalty
- Product company experience is heavily weighted
- Evidence of retrieval/search/recommendation system work in career descriptions
- Years of experience penalized outside 5-9 range (JD-specified)
- Irrelevant titles (Marketing, HR, Accountant) → 90% score penalty

**Availability (20%):**
- JD explicitly warns: "A perfect-on-paper candidate who hasn't logged in for 6 months... is not actually available"
- `last_active_date` recency: exponential decay after 6 months inactive
- `recruiter_response_rate`: low (<30%) is a major red flag
- `notice_period_days`: >90 days applies penalty (JD says sub-30 preferred)
- `open_to_work_flag`: hard boost
- `interview_completion_rate`: reliability signal

**Location (10%):**
- Target cities (Pune/Noida/Hyderabad/Mumbai/Delhi NCR/Bangalore): full score
- India + willing to relocate: 80%
- Outside India + not willing: 5%

### Honeypot Detection

Checks for 6 honeypot signatures:
1. Impossible tenure (worked at company longer than its existence)
2. Expert/advanced skills with 0 months duration (≥3 such skills)
3. Excessive zero-duration expert skills (≥5)
4. Large experience gap (claimed years vs career history sum)
5. Suspiciously perfect signals (4+ metrics at near-maximum)
6. Keyword stuffers (AI skills + non-AI title + no AI career work)

---

## Compute Constraints Satisfied

| Constraint | Limit | Actual |
|---|---|---|
| Runtime | ≤ 5 min | ~90 seconds |
| Memory | ≤ 16 GB | ~3-4 GB peak |
| GPU | Not allowed | CPU only |
| Network | Not allowed | All offline |

The pre-computation step (embedding 100k candidates) runs once in Colab
and takes ~10 minutes with a T4 GPU. This is NOT part of the ranking step.

---

## Sandbox Demo

🤗 **HuggingFace Spaces:** https://huggingface.co/spaces/YOUR_USERNAME/redrob-ranker

The sandbox accepts up to 100 candidates as JSON input and returns a
ranked CSV. Uses the same pipeline as the full submission.

---

## AI Tools Declaration

Used Claude for architecture discussion and code review. All engineering
decisions, feature weights, and scoring logic were designed and validated
by the team. No candidate data was sent to any external LLM.
