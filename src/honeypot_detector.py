"""
honeypot_detector.py — Detect honeypot and trap candidates.

From submission_spec.md Section 7:
  "The dataset contains ~80 honeypot candidates with subtly impossible profiles:
   e.g., 8 years of experience at a company founded 3 years ago; expert
   proficiency in 10 skills with 0 years used."

  "Submissions with honeypot rate > 10% in top 100 are DISQUALIFIED."

Honeypot types we detect:
  1. Impossible tenure: years at company > company's possible age
  2. Skill inflation: expert/advanced in many skills with 0 duration_months
  3. Experience mismatch: claimed years_of_experience vs career history sum
  4. Perfect-everything profile: suspiciously high scores on everything
  5. Keyword stuffers: skills list full of AI keywords but title/career mismatch
"""

from typing import Dict, Any, List, Tuple
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

# Reference date for all calculations
REFERENCE_DATE = date(2026, 6, 11)


def detect_honeypot(candidate: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Check if a candidate is a honeypot.
    
    Returns:
        (is_honeypot: bool, reasons: List[str])
    """
    reasons = []
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    # --- Check 1: Impossible tenure ---
    # If someone claims to have worked at a company for longer than the company
    # could possibly have existed based on their start date
    for job in career:
        start_str = job.get("start_date", "")
        duration = job.get("duration_months", 0)
        company = job.get("company", "")

        if start_str:
            try:
                start = date.fromisoformat(start_str)
                # If they started before 2000 but company size is 10001+
                # and duration is impossibly long
                months_possible = (
                    (REFERENCE_DATE.year - start.year) * 12
                    + (REFERENCE_DATE.month - start.month)
                )
                if duration > months_possible + 2:  # +2 for rounding tolerance
                    reasons.append(
                        f"Impossible tenure: {duration}mo at {company} "
                        f"but only {months_possible}mo since start date {start_str}"
                    )
            except (ValueError, TypeError):
                pass

    # --- Check 2: Skill inflation (expert + 0 months) ---
    # A candidate claiming expert-level proficiency with 0 months of use
    # is a clear honeypot signal
    suspicious_skills = []
    for skill in skills:
        name = skill.get("name", "")
        proficiency = skill.get("proficiency", "")
        duration = skill.get("duration_months", 0)

        if proficiency in ("expert", "advanced") and duration == 0:
            suspicious_skills.append(name)

    if len(suspicious_skills) >= 3:
        reasons.append(
            f"Skill inflation: {len(suspicious_skills)} expert/advanced skills "
            f"with 0 months experience: {suspicious_skills[:5]}"
        )

    # --- Check 3: Too many expert skills ---
    # The JD explicitly warns about "expert proficiency in 10 skills with 0 years used"
    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    if len(expert_skills) >= 8:
        zero_duration_experts = [
            s for s in expert_skills if s.get("duration_months", 0) == 0
        ]
        if len(zero_duration_experts) >= 5:
            reasons.append(
                f"Too many zero-duration expert skills: "
                f"{len(zero_duration_experts)} skills listed as expert with 0 months"
            )

    # --- Check 4: Career history sum vs claimed years_of_experience ---
    claimed_yoe = profile.get("years_of_experience", 0)
    career_months_total = sum(
        job.get("duration_months", 0) for job in career
    )
    career_years_total = career_months_total / 12

    # Allow up to 3 years gap (pre-career, gaps, etc.) but flag large discrepancies
    if claimed_yoe > career_years_total + 5 and claimed_yoe > 3:
        reasons.append(
            f"Experience mismatch: claims {claimed_yoe:.1f} years but "
            f"career history sums to only {career_years_total:.1f} years"
        )

    # --- Check 5: Suspicious signal pattern ---
    # All signals suspiciously perfect
    response_rate = signals.get("recruiter_response_rate", 0)
    interview_rate = signals.get("interview_completion_rate", 0)
    offer_rate = signals.get("offer_acceptance_rate", -1)
    completeness = signals.get("profile_completeness_score", 0)
    github = signals.get("github_activity_score", -1)

    perfect_count = sum([
        response_rate >= 0.99,
        interview_rate >= 0.99,
        offer_rate >= 0.99,
        completeness >= 99,
        github >= 99
    ])
    if perfect_count >= 4:
        reasons.append(
            f"Suspiciously perfect signals: {perfect_count}/5 metrics at near-maximum"
        )

    # --- Check 6: Keyword stuffer detection ---
    # Many AI keywords in skills but title/career shows no AI role
    ai_skill_names = {
        "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
        "elasticsearch", "sentence-transformers", "dense retrieval",
        "bm25", "retrieval", "embedding", "embeddings", "rag",
        "llm", "fine-tuning", "lora", "nlp", "pytorch", "transformers"
    }
    skill_names_lower = {s.get("name", "").lower() for s in skills}
    ai_skills_count = len(skill_names_lower & ai_skill_names)

    non_ai_titles = {
        "marketing", "sales", "hr", "human resources", "accountant",
        "operations", "customer support", "civil engineer", "mechanical engineer",
        "content writer", "graphic designer", "project manager",
        "business analyst", "product manager"
    }
    title_lower = profile.get("current_title", "").lower()
    is_non_ai_title = any(t in title_lower for t in non_ai_titles)

    # Check if career history descriptions mention AI/ML work
    career_desc_combined = " ".join(
        job.get("description", "").lower() for job in career
    )
    has_ai_career_work = any(
        kw in career_desc_combined
        for kw in ["retrieval", "embedding", "vector", "nlp", "machine learning",
                   "neural", "model", "ranking", "recommendation"]
    )

    if ai_skills_count >= 6 and is_non_ai_title and not has_ai_career_work:
        reasons.append(
            f"Keyword stuffer: {ai_skills_count} AI skills listed but "
            f"title is '{profile.get('current_title', '')}' with no AI career work"
        )

    is_honeypot = len(reasons) > 0
    return is_honeypot, reasons


def score_honeypot_risk(candidate: Dict[str, Any]) -> float:
    """
    Return a honeypot risk score between 0.0 (clean) and 1.0 (definite honeypot).
    Used as a multiplier to penalize suspicious candidates without hard-removing them.
    """
    is_honeypot, reasons = detect_honeypot(candidate)
    if not is_honeypot:
        return 0.0

    # Scale by number and severity of reasons
    base_risk = min(1.0, len(reasons) * 0.4)
    return base_risk


def filter_honeypots(
    candidates: List[Dict[str, Any]],
    verbose: bool = False
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split candidates into (clean, honeypots).
    
    Returns:
        (clean_candidates, honeypot_candidates)
    """
    clean = []
    honeypots = []

    for c in candidates:
        is_hp, reasons = detect_honeypot(c)
        if is_hp:
            honeypots.append(c)
            if verbose:
                print(
                    f"  HONEYPOT: {c['candidate_id']} | "
                    f"{c['profile']['current_title']} | {reasons[0]}"
                )
        else:
            clean.append(c)

    return clean, honeypots
