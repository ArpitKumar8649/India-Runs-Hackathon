"""
reasoner.py — Generate honest, specific, per-candidate reasoning strings.

From submission_spec.md Section 3 (Stage 4 manual review):
  - Reference specific facts from the candidate's profile
  - Connect to specific JD requirements (not generic praise)
  - Acknowledge obvious gaps/concerns where they exist
  - No hallucination — every claim must exist in the profile
  - Reasoning tone must match rank (rank-5 with critical reasoning = red flag)

Strategy: Build reasoning from actual FeatureVector sub-scores.
  - High rankers: lead with strongest positives, mention any minor concerns
  - Mid rankers: balanced view of strengths and gaps
  - Low rankers: lead with why they're in top-100 but acknowledge major gaps

Critically: we construct every sentence from actual profile data we can
verify — we never invent facts.
"""

from typing import Dict, Any, List
from datetime import date

from src.features import FeatureVector

REFERENCE_DATE = date(2026, 6, 11)


def generate_reasoning(
    candidate: Dict[str, Any],
    fv: FeatureVector,
    rank: int,
    final_score: float
) -> str:
    """
    Generate a 1-2 sentence reasoning for a candidate at a given rank.
    
    Uses actual profile fields — no hallucination.
    Tone varies by rank tier.
    
    Args:
        candidate:   Full candidate dict
        fv:          FeatureVector with all computed sub-scores
        rank:        Final rank (1-100)
        final_score: Final blended score
    
    Returns:
        1-2 sentence reasoning string.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    title = profile.get("current_title", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")

    # ── Build positive evidence ───────────────────────────────────────────

    positives = []
    concerns = []

    # Title + experience
    if not fv.is_disqualifier_title and not fv.is_consulting_only:
        positives.append(f"{title} with {yoe:.1f} years of experience")

    # Required skills found
    if fv.required_skills_found:
        top_skills = fv.required_skills_found[:3]
        positives.append(f"relevant skills in {', '.join(top_skills)}")

    # Product company experience
    if fv.has_product_company and not fv.is_consulting_only:
        # Find the product company name
        for job in career:
            company = job.get("company", "")
            is_consulting = any(
                cf in company.lower()
                for cf in {
                    "tcs", "infosys", "wipro", "accenture", "cognizant",
                    "capgemini", "hcl", "tech mahindra"
                }
            )
            if not is_consulting and job.get("duration_months", 0) > 12:
                positives.append(f"product company experience at {company}")
                break

    # Retrieval/search career work
    if fv.has_retrieval_career_work:
        positives.append("career history includes retrieval/search/ranking work")

    # Open to work + active
    if fv.open_to_work:
        positives.append("actively open to work")

    # GitHub activity
    if fv.github_score >= 50:
        positives.append(f"strong GitHub activity score ({fv.github_score:.0f}/100)")
    elif fv.github_score >= 25:
        positives.append(f"moderate GitHub activity ({fv.github_score:.0f}/100)")

    # Location
    if fv.in_target_location:
        positives.append(f"based in {location}")

    # ── Build concern evidence ─────────────────────────────────────────────

    # Notice period
    if fv.notice_period_days > 90:
        concerns.append(f"long notice period ({fv.notice_period_days} days)")
    elif fv.notice_period_days > 60:
        concerns.append(f"notice period of {fv.notice_period_days} days")

    # Recency
    if fv.days_since_active > 270:
        months_inactive = fv.days_since_active // 30
        concerns.append(f"last active {months_inactive} months ago")
    elif fv.days_since_active > 180:
        concerns.append("hasn't logged in for 6+ months")

    # Response rate
    if fv.response_rate < 0.30:
        concerns.append(f"low recruiter response rate ({fv.response_rate:.0%})")
    elif fv.response_rate < 0.50:
        concerns.append(f"moderate response rate ({fv.response_rate:.0%})")

    # Location concern
    if not fv.in_target_location and not fv.willing_to_relocate:
        country = profile.get("country", "")
        if country.lower() not in ("india", "in"):
            concerns.append(f"based outside India ({location}), not willing to relocate")
        else:
            concerns.append(f"not in target cities and not willing to relocate")

    # Consulting-only
    if fv.is_consulting_only:
        concerns.append("entire career at consulting/services companies")

    # No relevant skills
    if not fv.required_skills_found:
        concerns.append("no direct match on required AI/retrieval skills")

    # ── Build the sentence ────────────────────────────────────────────────

    if rank <= 10:
        return _build_top_tier_reasoning(positives, concerns, title, yoe, fv)
    elif rank <= 30:
        return _build_mid_high_reasoning(positives, concerns, title, yoe, fv)
    elif rank <= 60:
        return _build_mid_reasoning(positives, concerns, title, yoe, fv)
    else:
        return _build_low_reasoning(positives, concerns, title, yoe, fv, rank)


def _build_top_tier_reasoning(
    positives: List[str],
    concerns: List[str],
    title: str,
    yoe: float,
    fv: FeatureVector
) -> str:
    """Ranks 1-10: Lead with strongest signal, acknowledge any concern."""
    if not positives:
        return f"{title} with {yoe:.1f} years; included based on semantic match to JD."

    main = positives[0].capitalize()
    if len(positives) > 1:
        main += f"; {positives[1]}"

    sentence1 = main + "."

    if concerns:
        sentence2 = f"Note: {concerns[0]}."
        return f"{sentence1} {sentence2}"
    elif len(positives) > 2:
        return f"{sentence1} Also: {positives[2]}."
    return sentence1


def _build_mid_high_reasoning(
    positives: List[str],
    concerns: List[str],
    title: str,
    yoe: float,
    fv: FeatureVector
) -> str:
    """Ranks 11-30: Balanced, note the key strength and main gap."""
    strength = positives[0].capitalize() if positives else f"{title} with {yoe:.1f} years"
    concern = concerns[0] if concerns else None

    sentence1 = f"{strength}."
    if concern:
        return f"{sentence1} Main concern: {concern}."
    return sentence1


def _build_mid_reasoning(
    positives: List[str],
    concerns: List[str],
    title: str,
    yoe: float,
    fv: FeatureVector
) -> str:
    """Ranks 31-60: Honest about gaps, explain why still in top 100."""
    if positives and concerns:
        return (
            f"{positives[0].capitalize()}; however, {concerns[0]}. "
            f"Included based on partial match to JD requirements."
        )
    elif positives:
        return f"{positives[0].capitalize()}; skill or career gaps reduce ranking."
    else:
        return (
            f"{title}, {yoe:.1f} years — adjacent skills or experience; "
            f"below cutoff on key JD requirements but included in top 100."
        )


def _build_low_reasoning(
    positives: List[str],
    concerns: List[str],
    title: str,
    yoe: float,
    fv: FeatureVector,
    rank: int
) -> str:
    """Ranks 61-100: Honest — explain what got them in top 100 despite gaps."""
    if concerns:
        main_concern = concerns[0]
    else:
        main_concern = "limited match to core JD requirements"

    if positives:
        return (
            f"Included at rank {rank} due to {positives[0]}; "
            f"however, {main_concern} limits higher placement."
        )
    else:
        return (
            f"Adjacent skills only — {main_concern}; "
            f"included as ranked filler given available profile signals."
        )
