"""
features.py — Feature engineering for candidate scoring.

Builds a FeatureVector for each candidate based on:
  A) Skill Match Score    (35% weight)
  B) Career Fit Score     (35% weight)  
  C) Availability Score   (20% weight)
  D) Location Score       (10% weight)

Every score is normalized to [0.0, 1.0].

Design principle: The JD is very specific about what it wants AND what it
doesn't want. We encode both sides — positive signals boost the score,
explicit disqualifiers apply hard penalties.
"""

from typing import Dict, Any, List, Tuple
from datetime import date
import math

# Reference date for recency calculations
REFERENCE_DATE = date(2026, 6, 11)

# ── JD-derived constants ────────────────────────────────────────────────────

# Tier 1: Must-have skills (absolute requirements from JD)
REQUIRED_SKILLS = {
    "embeddings", "sentence-transformers", "sentence transformers",
    "vector search", "faiss", "pinecone", "weaviate", "qdrant",
    "milvus", "opensearch", "elasticsearch", "hybrid search",
    "dense retrieval", "retrieval", "python", "ranking",
    "information retrieval", "semantic search"
}

# Tier 2: Nice-to-have (bonus but not required)
BONUS_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "fine tuning",
    "learning to rank", "xgboost", "lightgbm", "reranking",
    "cross-encoder", "bi-encoder", "ndcg", "mrr", "map",
    "a/b testing", "rag", "retrieval augmented generation",
    "pytorch", "transformers", "huggingface", "hugging face",
    "nlp", "natural language processing", "llm"
}

# Tier 3: Explicit disqualifiers from JD
DISQUALIFIER_TITLES = {
    "marketing", "sales manager", "hr manager", "human resources",
    "accountant", "finance manager", "content writer", "graphic designer",
    "customer support", "civil engineer", "mechanical engineer",
    "operations manager", "supply chain"
}

# Consulting companies: JD explicitly says consulting-only = disqualify
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy services", "infosys", "wipro",
    "accenture", "cognizant", "capgemini", "hcl technologies",
    "hcl", "tech mahindra", "hexaware", "mphasis", "ltimindtree",
    "mindtree", "niit technologies", "syntel", "zensar",
    "igate", "firstsource", "wns global", "genpact"
}

# Target locations from JD
TARGET_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore",
    "bengaluru", "gurugram", "gurgaon", "ncr", "delhi ncr",
    "new delhi", "greater noida"
}

# JD says: experience 5-9 years, ideal 6-8 years
YOE_MIN = 5.0
YOE_IDEAL_LOW = 6.0
YOE_IDEAL_HIGH = 8.0
YOE_MAX = 9.0

# Notice period: sub-30 preferred, buyout up to 30, 30+ is penalty
NOTICE_IDEAL_MAX = 30
NOTICE_BUYOUT_MAX = 60
NOTICE_ACCEPTABLE_MAX = 90

# Availability: inactive for 6+ months = down-weight
ACTIVE_RECENCY_GOOD_DAYS = 90      # Active in last 3 months = good
ACTIVE_RECENCY_OK_DAYS = 180       # 3-6 months = ok
ACTIVE_RECENCY_BAD_DAYS = 180      # >6 months = penalize

PROFICIENCY_SCORE = {
    "expert": 1.0,
    "advanced": 0.75,
    "intermediate": 0.5,
    "beginner": 0.25
}


class FeatureVector:
    """Holds all computed features for a single candidate."""

    def __init__(self):
        # Component scores (all 0.0-1.0)
        self.skill_match_score: float = 0.0
        self.career_fit_score: float = 0.0
        self.availability_score: float = 0.0
        self.location_score: float = 0.0

        # Sub-features (for reasoning generation)
        self.required_skills_found: List[str] = []
        self.bonus_skills_found: List[str] = []
        self.years_of_experience: float = 0.0
        self.has_product_company: bool = False
        self.is_consulting_only: bool = False
        self.has_retrieval_career_work: bool = False
        self.notice_period_days: int = 0
        self.days_since_active: int = 0
        self.open_to_work: bool = False
        self.response_rate: float = 0.0
        self.github_score: float = 0.0
        self.interview_completion: float = 0.0
        self.in_target_location: bool = False
        self.willing_to_relocate: bool = False
        self.is_disqualifier_title: bool = False
        self.honeypot_risk: float = 0.0
        self.assessment_score: float = 0.0

        # Final combined score
        self.final_score: float = 0.0


def compute_features(candidate: Dict[str, Any]) -> FeatureVector:
    """
    Compute all features for a candidate.
    Returns a FeatureVector with all sub-scores populated.
    """
    fv = FeatureVector()

    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    # ── A. Skill Match Score ──────────────────────────────────────────────
    fv.skill_match_score, fv.required_skills_found, fv.bonus_skills_found, fv.assessment_score = (
        _compute_skill_score(skills, signals)
    )

    # ── B. Career Fit Score ───────────────────────────────────────────────
    (
        fv.career_fit_score,
        fv.years_of_experience,
        fv.has_product_company,
        fv.is_consulting_only,
        fv.has_retrieval_career_work,
        fv.is_disqualifier_title
    ) = _compute_career_score(profile, career)

    # ── C. Availability Score ─────────────────────────────────────────────
    (
        fv.availability_score,
        fv.notice_period_days,
        fv.days_since_active,
        fv.open_to_work,
        fv.response_rate,
        fv.github_score,
        fv.interview_completion
    ) = _compute_availability_score(signals)

    # ── D. Location Score ─────────────────────────────────────────────────
    fv.location_score, fv.in_target_location, fv.willing_to_relocate = (
        _compute_location_score(profile, signals)
    )

    # ── Final Score (weighted combination) ───────────────────────────────
    fv.final_score = _combine_scores(fv)

    return fv


def _compute_skill_score(
    skills: List[Dict],
    signals: Dict
) -> Tuple[float, List[str], List[str], float]:
    """
    Score skill match against JD requirements.
    
    Three components:
    1. Required skill coverage (how many must-have skills covered)
    2. Skill quality (proficiency + duration)
    3. Assessment scores from Redrob platform
    """
    skill_names_lower = {}
    for s in skills:
        name = s.get("name", "").lower()
        skill_names_lower[name] = s

    # --- Required skills ---
    required_found = []
    required_score = 0.0

    for req_skill in REQUIRED_SKILLS:
        # Check exact match or substring match
        matched = None
        for candidate_skill in skill_names_lower:
            if req_skill in candidate_skill or candidate_skill in req_skill:
                matched = candidate_skill
                break

        if matched:
            skill_data = skill_names_lower[matched]
            proficiency = skill_data.get("proficiency", "beginner")
            duration = skill_data.get("duration_months", 0)

            # Weight by proficiency
            prof_score = PROFICIENCY_SCORE.get(proficiency, 0.25)

            # Weight by duration (diminishing returns after 36 months)
            dur_score = min(1.0, duration / 36) if duration > 0 else 0.3

            skill_score = 0.6 * prof_score + 0.4 * dur_score
            required_score += skill_score
            required_found.append(f"{skill_data.get('name', matched)} ({proficiency})")

    # Normalize by number of required skills
    required_normalized = min(1.0, required_score / (len(REQUIRED_SKILLS) * 0.5))

    # --- Bonus skills ---
    bonus_found = []
    bonus_score = 0.0

    for bonus_skill in BONUS_SKILLS:
        for candidate_skill in skill_names_lower:
            if bonus_skill in candidate_skill or candidate_skill in bonus_skill:
                skill_data = skill_names_lower[candidate_skill]
                prof_score = PROFICIENCY_SCORE.get(
                    skill_data.get("proficiency", "beginner"), 0.25
                )
                bonus_score += prof_score * 0.5
                bonus_found.append(skill_data.get("name", candidate_skill))
                break

    bonus_normalized = min(1.0, bonus_score / (len(BONUS_SKILLS) * 0.3))

    # --- Assessment scores from Redrob platform ---
    assessment_scores = signals.get("skill_assessment_scores", {})
    relevant_assessments = []

    for skill_name, score in assessment_scores.items():
        skill_lower = skill_name.lower()
        is_relevant = any(
            req in skill_lower or skill_lower in req
            for req in (REQUIRED_SKILLS | BONUS_SKILLS)
        )
        if is_relevant:
            relevant_assessments.append(score)

    assessment_score = 0.0
    if relevant_assessments:
        assessment_score = sum(relevant_assessments) / len(relevant_assessments) / 100

    # Final skill score
    # Required coverage is most important, then bonus, then assessments
    final = (
        0.65 * required_normalized
        + 0.20 * bonus_normalized
        + 0.15 * assessment_score
    )

    return final, required_found[:8], bonus_found[:6], assessment_score


def _compute_career_score(
    profile: Dict,
    career: List[Dict]
) -> Tuple[float, float, bool, bool, bool, bool]:
    """
    Score career fit based on:
    - Years of experience (5-9 ideal range)
    - Product company vs consulting experience
    - Evidence of retrieval/search/recommendation work
    - Current title match
    - Company size (startup/mid-size preferred)
    """
    yoe = profile.get("years_of_experience", 0)
    current_title = profile.get("current_title", "").lower()

    # --- Years of experience ---
    if YOE_IDEAL_LOW <= yoe <= YOE_IDEAL_HIGH:
        yoe_score = 1.0
    elif YOE_MIN <= yoe < YOE_IDEAL_LOW:
        yoe_score = 0.7 + 0.3 * (yoe - YOE_MIN) / (YOE_IDEAL_LOW - YOE_MIN)
    elif YOE_IDEAL_HIGH < yoe <= YOE_MAX:
        yoe_score = 0.7 + 0.3 * (YOE_MAX - yoe) / (YOE_MAX - YOE_IDEAL_HIGH)
    elif yoe < YOE_MIN:
        yoe_score = max(0.1, yoe / YOE_MIN * 0.5)
    else:  # yoe > YOE_MAX (over 9 years)
        yoe_score = max(0.3, 1.0 - (yoe - YOE_MAX) * 0.05)

    # --- Title check ---
    is_disqualifier = any(t in current_title for t in DISQUALIFIER_TITLES)

    # Positive title signals
    positive_title_terms = {
        "ml engineer", "machine learning", "ai engineer", "data scientist",
        "nlp engineer", "research engineer", "applied scientist",
        "software engineer", "senior engineer", "staff engineer",
        "backend engineer", "platform engineer", "search engineer",
        "recommendation", "retrieval"
    }
    title_score = 0.5  # Neutral default
    if any(t in current_title for t in positive_title_terms):
        title_score = 1.0
    elif is_disqualifier:
        title_score = 0.0

    # --- Company type analysis ---
    total_months = 0
    consulting_months = 0
    product_months = 0
    has_product_company = False
    is_consulting_only = False

    for job in career:
        company = job.get("company", "").lower()
        duration = job.get("duration_months", 0)
        total_months += duration

        is_consulting = any(cf in company for cf in CONSULTING_COMPANIES)
        if is_consulting:
            consulting_months += duration
        else:
            product_months += duration
            has_product_company = True

    if total_months > 0:
        consulting_ratio = consulting_months / total_months
        product_ratio = product_months / total_months
        # JD says consulting-ONLY career is disqualifier
        is_consulting_only = consulting_ratio > 0.9 and total_months > 24
    else:
        consulting_ratio = 0
        product_ratio = 0

    company_type_score = product_ratio if total_months > 0 else 0.5

    # --- Evidence of retrieval/search/recommendation work ---
    career_text = " ".join(
        job.get("description", "").lower() + " " + job.get("title", "").lower()
        for job in career
    )

    retrieval_keywords = {
        "retrieval", "search", "ranking", "recommendation", "embedding",
        "vector", "faiss", "similarity", "nlp", "information retrieval",
        "semantic", "matching", "candidate ranking", "re-rank", "rerank"
    }
    retrieval_hits = sum(1 for kw in retrieval_keywords if kw in career_text)
    has_retrieval_work = retrieval_hits >= 2

    retrieval_score = min(1.0, retrieval_hits / 4)

    # --- Company size (startups/mid-size preferred per JD) ---
    company_sizes = []
    for job in career:
        if job.get("is_current", False):
            company_sizes.append(job.get("company_size", ""))

    size_score = 0.5  # Default neutral
    if company_sizes:
        size = company_sizes[0]
        size_scores = {
            "1-10": 1.0, "11-50": 1.0, "51-200": 0.9, "201-500": 0.85,
            "501-1000": 0.75, "1001-5000": 0.6, "5001-10000": 0.4, "10001+": 0.3
        }
        size_score = size_scores.get(size, 0.5)

    # --- Combine career sub-scores ---
    if is_consulting_only:
        # Hard penalty for consulting-only (JD explicitly disqualifies)
        career_score = 0.1
    elif is_disqualifier:
        # Hard penalty for irrelevant title
        career_score = 0.05
    else:
        career_score = (
            0.25 * yoe_score
            + 0.30 * title_score
            + 0.20 * company_type_score
            + 0.20 * retrieval_score
            + 0.05 * size_score
        )

    return (
        min(1.0, career_score),
        yoe,
        has_product_company,
        is_consulting_only,
        has_retrieval_work,
        is_disqualifier
    )


def _compute_availability_score(
    signals: Dict
) -> Tuple[float, int, int, bool, float, float, float]:
    """
    Score candidate availability using Redrob behavioral signals.
    
    Key insight from JD:
    "A perfect-on-paper candidate who hasn't logged in for 6 months and has
    a 5% recruiter response rate is, for hiring purposes, not actually available."
    """
    # Extract signal values
    open_to_work = signals.get("open_to_work_flag", False)
    last_active_str = signals.get("last_active_date", "")
    notice_period = signals.get("notice_period_days", 90)
    response_rate = signals.get("recruiter_response_rate", 0.0)
    avg_response_hours = signals.get("avg_response_time_hours", 168)
    github_score = signals.get("github_activity_score", -1)
    interview_rate = signals.get("interview_completion_rate", 0.5)
    offer_rate = signals.get("offer_acceptance_rate", -1)
    completeness = signals.get("profile_completeness_score", 0)
    applications_30d = signals.get("applications_submitted_30d", 0)
    saved_30d = signals.get("saved_by_recruiters_30d", 0)

    # --- Last active recency ---
    days_since_active = 999
    if last_active_str:
        try:
            last_active = date.fromisoformat(last_active_str)
            days_since_active = (REFERENCE_DATE - last_active).days
        except (ValueError, TypeError):
            pass

    if days_since_active <= ACTIVE_RECENCY_GOOD_DAYS:
        recency_score = 1.0
    elif days_since_active <= ACTIVE_RECENCY_OK_DAYS:
        recency_score = 0.6
    else:
        # Exponential decay after 6 months inactive
        recency_score = max(0.0, 0.6 * math.exp(
            -(days_since_active - ACTIVE_RECENCY_OK_DAYS) / 180
        ))

    # --- Open to work signal ---
    open_score = 1.0 if open_to_work else 0.3
    # Boost if actively applying
    if applications_30d >= 3:
        open_score = min(1.0, open_score + 0.2)

    # --- Notice period ---
    if notice_period <= NOTICE_IDEAL_MAX:
        notice_score = 1.0
    elif notice_period <= NOTICE_BUYOUT_MAX:
        notice_score = 0.8  # Still in scope (they'll buy out up to 30 days)
    elif notice_period <= NOTICE_ACCEPTABLE_MAX:
        notice_score = 0.5  # "bar gets higher"
    else:
        notice_score = max(0.1, 1.0 - (notice_period - 90) / 180)

    # --- Response rate (critical: low = not actually available) ---
    if response_rate >= 0.7:
        response_score = 1.0
    elif response_rate >= 0.4:
        response_score = 0.6 + 0.4 * (response_rate - 0.4) / 0.3
    else:
        # Low response rate is a major red flag
        response_score = response_rate / 0.4 * 0.6

    # Factor in response time
    if avg_response_hours <= 24:
        response_time_bonus = 0.2
    elif avg_response_hours <= 72:
        response_time_bonus = 0.1
    else:
        response_time_bonus = 0.0

    response_score = min(1.0, response_score + response_time_bonus)

    # --- GitHub activity (JD says it cares about code quality and open source) ---
    if github_score == -1:
        # No GitHub linked — neutral, not penalized
        github_norm = 0.4
    elif github_score >= 70:
        github_norm = 1.0
    elif github_score >= 40:
        github_norm = 0.7
    elif github_score >= 15:
        github_norm = 0.5
    else:
        github_norm = 0.2 + github_score / 15 * 0.3

    # --- Interview completion (reliability signal) ---
    interview_score = interview_rate if interview_rate >= 0 else 0.5

    # --- Profile completeness ---
    completeness_score = completeness / 100

    # --- Saved by recruiters (social proof) ---
    social_proof = min(1.0, saved_30d / 10)

    # --- Combine availability sub-scores ---
    # Recency and response rate are most critical (JD explicitly mentions both)
    availability_score = (
        0.25 * recency_score
        + 0.20 * open_score
        + 0.20 * notice_score
        + 0.15 * response_score
        + 0.10 * github_norm
        + 0.05 * interview_score
        + 0.03 * completeness_score
        + 0.02 * social_proof
    )

    return (
        min(1.0, availability_score),
        notice_period,
        days_since_active,
        open_to_work,
        response_rate,
        github_score if github_score >= 0 else 0.0,
        interview_rate
    )


def _compute_location_score(
    profile: Dict,
    signals: Dict
) -> Tuple[float, bool, bool]:
    """
    Score location fit.
    JD target: Pune, Noida, Hyderabad, Mumbai, Delhi NCR, Bangalore.
    Open to relocating candidates from Tier-1 Indian cities.
    Outside India: case-by-case, don't sponsor visas.
    """
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing_to_relocate = signals.get("willing_to_relocate", False)

    in_target = any(city in location for city in TARGET_CITIES)

    if in_target:
        loc_score = 1.0
    elif country in ("india", "in") or "india" in country:
        # In India but not target city
        if willing_to_relocate:
            loc_score = 0.8
        else:
            loc_score = 0.4
    else:
        # Outside India (case-by-case per JD, no visa sponsorship)
        if willing_to_relocate:
            loc_score = 0.3
        else:
            loc_score = 0.05

    return loc_score, in_target, willing_to_relocate


def _combine_scores(fv: FeatureVector) -> float:
    """
    Combine all component scores into a final score.
    
    Weights derived from JD emphasis:
    - Skill match: 35% (technical depth is crucial)
    - Career fit: 35% (product company + retrieval work is crucial)
    - Availability: 20% (behavioral signals matter a lot)
    - Location: 10% (preferred but flexible)
    """
    base_score = (
        0.35 * fv.skill_match_score
        + 0.35 * fv.career_fit_score
        + 0.20 * fv.availability_score
        + 0.10 * fv.location_score
    )

    # Hard penalties for explicit disqualifiers
    if fv.is_consulting_only:
        base_score *= 0.15  # Consulting-only career → near-zero
    elif fv.is_disqualifier_title:
        base_score *= 0.10  # Completely wrong title → near-zero

    # Penalty for very low availability
    if fv.days_since_active > 365:
        base_score *= 0.5  # 1+ year inactive: significant penalty

    # Honeypot penalty
    if fv.honeypot_risk > 0:
        base_score *= (1.0 - fv.honeypot_risk * 0.9)

    return min(1.0, max(0.0, base_score))
