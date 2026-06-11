"""
text_builder.py — Build rich text representations of candidates for embedding.

The quality of your FAISS retrieval depends directly on the quality of the
text you feed to the encoder. Naive approach: just dump all fields as a string.
Better approach: weight important fields, structure the text semantically,
and help the model understand what matters for this JD.

Key insight from the JD:
- Career descriptions are more important than skill lists
- Product company experience > consulting company experience  
- Title matters (ML Engineer >> Marketing Manager, even with identical skills)
- Skills with duration and proficiency > bare skill names
"""

from typing import Dict, Any, List


# JD-specific skill keywords — used to boost relevant skills in text
CORE_AI_SKILLS = {
    "embeddings", "vector search", "faiss", "pinecone", "weaviate", "qdrant",
    "milvus", "opensearch", "elasticsearch", "sentence-transformers", "sentence transformers",
    "dense retrieval", "hybrid search", "bm25", "retrieval", "ranking",
    "learning to rank", "reranking", "cross-encoder", "bi-encoder",
    "pytorch", "transformers", "hugging face", "huggingface",
    "llm", "llms", "large language model", "rag", "retrieval augmented generation",
    "fine-tuning", "fine tuning", "lora", "qlora", "peft",
    "nlp", "natural language processing", "text embeddings",
    "ndcg", "mrr", "map", "evaluation", "a/b testing",
    "python", "machine learning", "deep learning",
    "recommendation system", "search system", "information retrieval",
    "xgboost", "lightgbm", "neural ranking"
}

# Consulting-only companies (JD explicitly disqualifies consulting-only careers)
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl technologies", "hcl", "tech mahindra",
    "hexaware", "mphasis", "l&t infotech", "ltimindtree", "mindtree",
    "niit technologies", "syntel", "patni", "mastech", "zensar",
    "igate", "merilytics", "firstsource", "wns global"
}

# Target locations from the JD
TARGET_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore",
    "bengaluru", "gurugram", "gurgaon", "ncr", "delhi ncr"
}

PROFICIENCY_WEIGHT = {
    "expert": 4,
    "advanced": 3,
    "intermediate": 2,
    "beginner": 1
}


def build_candidate_text(candidate: Dict[str, Any]) -> str:
    """
    Build a rich, structured text string for a candidate.
    This text is what gets embedded by SentenceTransformer.

    Strategy:
    1. Lead with title and headline (high signal)
    2. Include career descriptions (most semantically rich)
    3. Add skills with proficiency (not just names)
    4. Note company types (product vs consulting)
    5. Include certifications and education field
    """
    parts = []

    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    certifications = candidate.get("certifications", [])

    # --- 1. Title + Headline (strongest signal) ---
    title = profile.get("current_title", "")
    headline = profile.get("headline", "")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "")

    parts.append(f"Title: {title}. Experience: {yoe:.1f} years. Location: {location}.")
    if headline:
        parts.append(f"Headline: {headline}.")

    # --- 2. Professional summary ---
    summary = profile.get("summary", "")
    if summary:
        parts.append(f"Summary: {summary}")

    # --- 3. Career history (most semantically rich) ---
    career_parts = []
    for job in career:
        job_title = job.get("title", "")
        company = job.get("company", "")
        industry = job.get("industry", "")
        duration = job.get("duration_months", 0)
        description = job.get("description", "")
        company_size = job.get("company_size", "")
        is_current = job.get("is_current", False)

        # Flag consulting vs product
        is_consulting = any(
            cf in company.lower() for cf in CONSULTING_COMPANIES
        )
        company_type = "consulting company" if is_consulting else "product company"

        job_text = (
            f"{'Current role' if is_current else 'Previous role'}: "
            f"{job_title} at {company} ({company_type}, {industry}, "
            f"{company_size} employees, {duration} months). {description}"
        )
        career_parts.append(job_text)

    if career_parts:
        parts.append("Career history: " + " | ".join(career_parts))

    # --- 4. Skills (weighted by proficiency and relevance) ---
    skill_strs = []
    core_skills_found = []

    for skill in skills:
        name = skill.get("name", "")
        proficiency = skill.get("proficiency", "beginner")
        duration = skill.get("duration_months", 0)
        endorsements = skill.get("endorsements", 0)

        # Check if this is a core AI/IR skill
        is_core = any(
            kw in name.lower() for kw in CORE_AI_SKILLS
        )

        # Build skill description
        skill_desc = f"{name} ({proficiency}"
        if duration > 0:
            years = duration / 12
            skill_desc += f", {years:.1f} years"
        if endorsements > 10:
            skill_desc += f", {endorsements} endorsements"
        skill_desc += ")"

        if is_core:
            core_skills_found.append(skill_desc)
        else:
            skill_strs.append(skill_desc)

    # Put core skills first for higher embedding weight
    all_skills = core_skills_found + skill_strs
    if all_skills:
        if core_skills_found:
            parts.append(
                f"Key AI/ML skills: {', '.join(core_skills_found)}."
            )
        if skill_strs:
            parts.append(f"Other skills: {', '.join(skill_strs[:10])}.")

    # --- 5. Education ---
    for edu in education:
        field = edu.get("field_of_study", "")
        degree = edu.get("degree", "")
        institution = edu.get("institution", "")
        tier = edu.get("tier", "")
        if field or degree:
            parts.append(
                f"Education: {degree} in {field} from {institution} ({tier})."
            )

    # --- 6. Certifications ---
    if certifications:
        cert_names = [c.get("name", "") for c in certifications[:5]]
        parts.append(f"Certifications: {', '.join(cert_names)}.")

    return " ".join(parts)


def build_jd_query_text() -> str:
    """
    Build the query text from the Job Description.
    
    This is what we embed to search against the candidate index.
    Critical: this should describe the IDEAL candidate, not just the JD text.
    We expand with synonyms and related concepts.
    """
    return (
        "Senior AI Engineer with 5 to 9 years of experience at product companies. "
        "Expert in embeddings-based retrieval systems using sentence-transformers, "
        "FAISS, Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch. "
        "Production experience with vector databases and hybrid search infrastructure. "
        "Strong Python skills. Experience designing evaluation frameworks for ranking "
        "systems using NDCG, MRR, MAP, A/B testing. "
        "Shipped end-to-end search, ranking, or recommendation systems to real users. "
        "Experience with LLM fine-tuning using LoRA, QLoRA, PEFT. "
        "Learning-to-rank models with XGBoost or LightGBM. "
        "Located in Pune, Noida, Hyderabad, Mumbai, Delhi NCR, Bangalore. "
        "Startup or product company background. "
        "Not from pure research or pure consulting background. "
        "Has worked on information retrieval, semantic search, dense retrieval. "
        "NLP and natural language processing background. "
        "RAG retrieval augmented generation experience. "
        "Hands-on coder who ships production ML systems."
    )
