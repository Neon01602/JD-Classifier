"""
Feature extraction for a single candidate record.

All functions here are pure and operate on one candidate dict (already
parsed from JSONL). No I/O, no network. This module is the single source
of truth for both the score and the reasoning generator, so reasoning
text is guaranteed to reflect the same evidence that drove the score.
"""

from __future__ import annotations

import re
import datetime
from dataclasses import dataclass, field

from taxonomy import (
    title_tier_and_score,
    CORE_SKILLS,
    CV_SPEECH_SKILLS,
    CONSULTING_FIRM_MARKERS,
    CONSULTING_INDUSTRIES,
    PREFERRED_CITIES,
    WELCOME_CITIES,
    LOCATION_SCORES,
)

TODAY = datetime.date(2026, 6, 14)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str | None) -> datetime.date | None:
    if not s:
        return None
    try:
        return datetime.date.fromisoformat(s)
    except ValueError:
        return None


def _days_since(date_str: str | None) -> int | None:
    d = _parse_date(date_str)
    if d is None:
        return None
    return (TODAY - d).days


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CandidateFeatures:
    candidate_id: str

    # Component scores, each in [0, 1] (before weighting)
    title_score: float = 0.0
    title_tier: str = "?"
    skills_score: float = 0.0
    experience_score: float = 0.0
    location_score: float = 0.0
    education_score: float = 0.0
    behavioral_multiplier: float = 1.0
    tiebreak_score: float = 0.0

    # Honeypot
    honeypot_flags: list[str] = field(default_factory=list)
    is_honeypot: bool = False

    # Evidence for reasoning (kept human-readable, not raw dumps)
    evidence: dict = field(default_factory=dict)

    # Final composite (filled in by scoring.py)
    final_score: float = 0.0


# ---------------------------------------------------------------------------
# Component extractors
# ---------------------------------------------------------------------------

def score_title(profile: dict, career_history: list[dict]) -> tuple[float, str, dict]:
    """Title-trajectory match. Decisive signal per JD final note.

    Boosts the base title-tier score if recent career history descriptions
    contain retrieval/ranking/ML production language even when the current
    title itself is generic (the "Tier-5 without buzzwords" case).
    Penalizes if the title looks senior/AI but the most recent role
    description shows no hands-on technical work (architecture/manager
    drift, per JD's "hasn't written code in 18 months" disqualifier).
    """
    title = profile["current_title"]
    tier, base = title_tier_and_score(title)

    # Recent career-history language signal (last 2 roles only).
    recent = career_history[:2]
    text = " ".join(h.get("description", "") for h in recent).lower()

    rescue_terms = [
        "retrieval", "ranking", "recommendation", "search", "embedding",
        "vector", "matching pipeline", "relevance", "feature pipeline",
        "evaluation harness", "ndcg", "a/b test", "hybrid retrieval",
    ]
    rescue_hits = sum(1 for term in rescue_terms if term in text)

    drift_terms = [
        "managed a team", "oversaw the roadmap", "stakeholder", "no longer write",
        "moved into management", "purely architectural", "vision and strategy",
    ]
    drift_hits = sum(1 for term in drift_terms if term in text)

    score = base
    note = None

    if tier in ("D", "E") and rescue_hits >= 2:
        # Generic/unrelated title, but career history shows real
        # retrieval/ranking production work -> rescue toward Tier C territory.
        score = min(0.65, base + 0.30 + 0.05 * rescue_hits)
        note = "career_history_rescue"
    elif tier in ("A", "B") and drift_hits >= 1 and rescue_hits == 0:
        # Senior/AI title but recent roles read as pure management/architecture.
        score = max(0.30, base - 0.35)
        note = "seniority_drift_penalty"

    return score, tier, {
        "current_title": title,
        "title_note": note,
        "recent_role_signal_hits": rescue_hits,
    }


def score_skills(skills: list[dict]) -> tuple[float, dict]:
    """Skills relevance with endorsement/duration trust multiplier.

    For each skill in the JD's core stack, contribution =
        category_weight * proficiency_weight * trust_multiplier
    trust_multiplier discounts skills with 0 endorsements AND 0 duration
    (keyword-stuffing signal).
    """
    PROF_WEIGHT = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}

    matched = []
    cv_only_hits = 0
    nlp_hits = 0
    total = 0.0
    max_possible = 0.0

    for s in skills:
        name = s["name"].strip().lower()
        prof = PROF_WEIGHT.get(s.get("proficiency", "intermediate"), 0.5)
        endorsements = s.get("endorsements", 0)
        duration = s.get("duration_months", 0)

        if name in CV_SPEECH_SKILLS:
            cv_only_hits += 1
        if name in ("nlp", "natural language processing", "information retrieval",
                     "information retrieval systems", "ranking systems",
                     "learning to rank"):
            nlp_hits += 1

        if name not in CORE_SKILLS:
            continue

        category, weight = CORE_SKILLS[name]

        trust = 1.0
        if endorsements == 0 and duration == 0:
            trust = 0.25  # likely keyword-stuffed
        elif endorsements == 0 or duration < 6:
            trust = 0.6

        contribution = weight * prof * trust
        total += contribution
        max_possible += weight  # cap at full weight per skill if perfectly trusted+expert

        if contribution > 0.3:
            matched.append((s["name"], s.get("proficiency"), endorsements, duration))

    # Normalize: cap the raw total at a reasonable max so a candidate with
    # many strong, trusted core skills approaches 1.0.
    NORMALIZER = 6.0  # ~6 weight-1.0 expert/trusted skills = perfect score
    skills_score = min(1.0, total / NORMALIZER)

    # CV/speech-only penalty: lots of CV/speech skills with no NLP/IR presence
    cv_only_penalty = 0.0
    if cv_only_hits >= 3 and nlp_hits == 0:
        cv_only_penalty = 0.20
        skills_score = max(0.0, skills_score - cv_only_penalty)

    matched.sort(key=lambda x: -x[2])  # by endorsements desc
    return skills_score, {
        "matched_core_skills": matched[:5],
        "cv_speech_only_penalty_applied": cv_only_penalty > 0,
    }


def score_experience(profile: dict, career_history: list[dict]) -> tuple[float, dict]:
    """Experience/seniority fit centered on the JD's 5-9yr soft band.

    Also penalizes pure-research-only career history (every role's
    industry/title reads as academic research with no production
    deployment signal) per JD disqualifier #1.
    """
    yoe = profile.get("years_of_experience", 0.0)

    # Triangular fit curve centered on 7, full score 5-9, tapering outside.
    if 5.0 <= yoe <= 9.0:
        fit = 1.0
    elif yoe < 5.0:
        fit = max(0.0, 1.0 - (5.0 - yoe) * 0.18)
    else:
        fit = max(0.0, 1.0 - (yoe - 9.0) * 0.12)

    # Pure-research-only check: every role description reads academic
    # (no "production", "shipped", "users", "scale", etc.) and titles
    # contain "research"/"scientist" without product-company signal.
    text_all = " ".join(h.get("description", "") for h in career_history).lower()
    research_terms = ["research", "publication", "academic", "lab", "thesis", "paper"]
    production_terms = ["production", "shipped", "users", "scale", "deployed",
                         "real-time", "latency", "throughput", "rollout"]
    research_hits = sum(1 for t in research_terms if t in text_all)
    production_hits = sum(1 for t in production_terms if t in text_all)

    research_only_penalty = 0.0
    if research_hits >= 2 and production_hits == 0:
        research_only_penalty = 0.5
        fit = max(0.0, fit - research_only_penalty)

    return fit, {
        "years_of_experience": yoe,
        "research_only_penalty_applied": research_only_penalty > 0,
    }


def score_location(profile: dict) -> tuple[float, dict]:
    location = profile.get("location", "")
    country = profile.get("country", "")
    loc_lower = location.lower()

    is_india = country.strip().lower() == "india"

    if is_india:
        if any(c in loc_lower for c in PREFERRED_CITIES):
            score = LOCATION_SCORES["preferred"]
            bucket = "preferred"
        elif any(c in loc_lower for c in WELCOME_CITIES):
            score = LOCATION_SCORES["welcome"]
            bucket = "welcome"
        else:
            score = LOCATION_SCORES["other_india"]
            bucket = "other_india"
    else:
        bucket = "outside_india"
        score = LOCATION_SCORES["outside_india_no_relocate"]  # default; relocate bonus applied later

    return score, {"location": location, "country": country, "location_bucket": bucket}


def score_education(education: list[dict], current_company: str, current_industry: str) -> tuple[float, dict]:
    """Light-weight education + product-vs-services company signal."""
    tier_scores = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.3, "unknown": 0.4}
    edu_score = 0.4
    best_tier = "unknown"
    if education:
        best = max(education, key=lambda e: tier_scores.get(e.get("tier", "unknown"), 0.4))
        best_tier = best.get("tier", "unknown")
        edu_score = tier_scores.get(best_tier, 0.4)

    company_lower = current_company.lower()
    industry_lower = current_industry.lower()
    is_consulting = (
        any(marker in company_lower for marker in CONSULTING_FIRM_MARKERS)
        or industry_lower in CONSULTING_INDUSTRIES
    )

    # Blend: education 60%, product-company bonus/penalty 40%.
    company_component = 0.3 if is_consulting else 0.8
    combined = 0.6 * edu_score + 0.4 * company_component

    return combined, {
        "education_tier": best_tier,
        "is_consulting_firm": is_consulting,
        "current_company": current_company,
    }


def score_behavioral(signals: dict) -> tuple[float, dict]:
    """Multiplicative modifier in roughly [0.4, 1.1] based on availability
    and engagement signals. This is what sinks 'ghost' candidates.
    """
    open_to_work = signals.get("open_to_work_flag", False)
    last_active_days = _days_since(signals.get("last_active_date"))
    resp_rate = signals.get("recruiter_response_rate", 0.0)
    interview_rate = signals.get("interview_completion_rate", 0.0)
    verified_email = signals.get("verified_email", False)
    verified_phone = signals.get("verified_phone", False)

    mult = 1.0

    # Recency of activity
    if last_active_days is not None:
        if last_active_days <= 14:
            mult *= 1.05
        elif last_active_days <= 45:
            mult *= 1.0
        elif last_active_days <= 120:
            mult *= 0.85
        else:
            mult *= 0.55  # ~6 months+ inactive -> "for hiring purposes, not available"

    # Open to work
    mult *= 1.05 if open_to_work else 0.80

    # Recruiter response rate
    if resp_rate >= 0.6:
        mult *= 1.05
    elif resp_rate >= 0.3:
        mult *= 1.0
    else:
        mult *= 0.85

    # Interview completion (only meaningful if they have interview history)
    if interview_rate < 0.5:
        mult *= 0.92

    # Verification — small trust signal
    if not (verified_email and verified_phone):
        mult *= 0.97

    return mult, {
        "last_active_days_ago": last_active_days,
        "open_to_work_flag": open_to_work,
        "recruiter_response_rate": resp_rate,
    }


def score_tiebreak(signals: dict) -> float:
    """Small continuous signal in [0, 1] used purely to break ties between
    candidates with identical rounded composite scores. Uses Redrob's own
    quality signals (skill assessment scores, profile completeness,
    endorsements, search/recruiter interest) so it remains within the
    "behavioral signal" spirit of the JD without double-counting heavily.
    """
    assess = signals.get("skill_assessment_scores", {}) or {}
    avg_assess = (sum(assess.values()) / len(assess) / 100.0) if assess else 0.5
    completeness = signals.get("profile_completeness_score", 50) / 100.0
    saved = min(signals.get("saved_by_recruiters_30d", 0), 20) / 20.0
    searches = min(signals.get("search_appearance_30d", 0), 50) / 50.0

    return 0.4 * avg_assess + 0.3 * completeness + 0.15 * saved + 0.15 * searches


# ---------------------------------------------------------------------------
# Honeypot detection
# ---------------------------------------------------------------------------

YOE_PATTERN = re.compile(r"(\d+\.?\d*)\s*years?\s*of")


def detect_honeypot(profile: dict, career_history: list[dict], skills: list[dict]) -> list[str]:
    flags = []
    yoe = profile.get("years_of_experience", 0.0)

    # 1. Summary-stated years vs. profile years_of_experience mismatch
    m = YOE_PATTERN.search(profile.get("summary", ""))
    if m:
        stated = float(m.group(1))
        if abs(stated - yoe) > 1.0:
            flags.append("yoe_summary_mismatch")

    # 2. Career span shorter than claimed years of experience
    starts = [h.get("start_date") for h in career_history if h.get("start_date")]
    if starts:
        earliest = min(starts)
        try:
            earliest_year = int(earliest[:4])
            span = TODAY.year - earliest_year
            if span < yoe - 2:
                flags.append("career_span_shorter_than_yoe")
        except ValueError:
            pass

    # 3. "Expert" proficiency with 0 duration_months on multiple skills
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    if expert_zero >= 2:
        flags.append("multiple_expert_zero_duration_skills")

    # 4. Implausible total months across overlapping/duplicated roles
    #    (e.g., duration_months sum vastly exceeds yoe*12 with no overlap reason)
    total_months = sum(h.get("duration_months", 0) for h in career_history)
    if total_months > (yoe * 12 + 18) and len(career_history) > 1:
        flags.append("career_duration_exceeds_total_experience")

    return flags


# ---------------------------------------------------------------------------
# Top-level extraction
# ---------------------------------------------------------------------------

def extract_features(candidate: dict) -> CandidateFeatures:
    profile = candidate["profile"]
    career_history = candidate.get("career_history", [])
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    feat = CandidateFeatures(candidate_id=candidate["candidate_id"])

    feat.title_score, feat.title_tier, title_ev = score_title(profile, career_history)
    feat.skills_score, skills_ev = score_skills(skills)
    feat.experience_score, exp_ev = score_experience(profile, career_history)
    feat.location_score, loc_ev = score_location(profile)
    feat.education_score, edu_ev = score_education(
        education, profile.get("current_company", ""), profile.get("current_industry", "")
    )
    feat.behavioral_multiplier, behav_ev = score_behavioral(signals)
    feat.tiebreak_score = score_tiebreak(signals)

    # Relocation bonus for outside-India candidates
    if loc_ev["location_bucket"] == "outside_india" and signals.get("willing_to_relocate"):
        feat.location_score = LOCATION_SCORES["outside_india_relocate"]
        loc_ev["location_bucket"] = "outside_india_relocate"

    feat.honeypot_flags = detect_honeypot(profile, career_history, skills)
    feat.is_honeypot = len(feat.honeypot_flags) >= 2

    feat.evidence = {
        **title_ev,
        **skills_ev,
        **exp_ev,
        **loc_ev,
        **edu_ev,
        **behav_ev,
        "honeypot_flags": feat.honeypot_flags,
    }

    return feat
