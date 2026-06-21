"""
Reasoning string generator.

Builds a 1-2 sentence reasoning string DIRECTLY from the same
CandidateFeatures object used for scoring, so:
  - facts referenced are guaranteed to exist in the profile (no hallucination)
  - the strongest and weakest components are named -> JD-connected,
    honest about concerns
  - tone follows from final_score -> rank-consistent
  - phrasing varies based on which components are strong/weak -> not templated
"""

from __future__ import annotations

import random

from features import CandidateFeatures

COMPONENT_LABELS = {
    "title": "current role/title alignment",
    "skills": "core retrieval/ranking skill match",
    "experience": "experience-level fit",
    "location": "location/relocation fit",
    "education": "education and company background",
}


def _component_scores(feat: CandidateFeatures) -> dict:
    return {
        "title": feat.title_score,
        "skills": feat.skills_score,
        "experience": feat.experience_score,
        "location": feat.location_score,
        "education": feat.education_score,
    }


def _format_skills_phrase(feat: CandidateFeatures) -> str | None:
    matched = feat.evidence.get("matched_core_skills", [])
    if not matched:
        return None
    names = [m[0] for m in matched[:3]]
    return ", ".join(names)


def _strength_phrase(feat: CandidateFeatures, key: str) -> str:
    ev = feat.evidence
    title = ev.get("current_title", "their current role")
    yoe = ev.get("years_of_experience")

    if key == "title":
        if feat.title_tier in ("A", "B"):
            return f"their title ({title}) is a direct/close match for the AI Engineer role"
        if ev.get("title_note") == "career_history_rescue":
            return (
                f"despite a {title} title, their recent role descriptions show "
                f"hands-on retrieval/ranking production work"
            )
        return f"their current role ({title}) shows relevant trajectory"

    if key == "skills":
        skills_phrase = _format_skills_phrase(feat)
        if skills_phrase:
            return f"they hold core retrieval/ranking skills ({skills_phrase})"
        return "their skill profile shows some relevant tooling"

    if key == "experience":
        return f"{yoe:.1f} years of experience sits squarely in the JD's 5-9 year band"

    if key == "location":
        bucket = ev.get("location_bucket")
        loc = ev.get("location")
        if bucket == "preferred":
            return f"they're based in {loc}, matching the JD's preferred locations"
        if bucket == "welcome":
            return f"they're based in {loc}, one of the JD's welcomed cities"
        return f"located in {loc}"

    if key == "education":
        if not ev.get("is_consulting_firm"):
            return f"current employer ({ev.get('current_company')}) is a product company, not a pure-services firm"
        return "education background is on file"

    return "this aspect is solid"


def _concern_phrase(feat: CandidateFeatures, key: str) -> str:
    ev = feat.evidence
    title = ev.get("current_title", "their current role")
    yoe = ev.get("years_of_experience")

    if key == "title":
        if ev.get("title_note") == "seniority_drift_penalty":
            return f"recent role descriptions read as management/architecture rather than hands-on coding"
        return f"current title ({title}) doesn't map closely onto the AI Engineer role"

    if key == "skills":
        if ev.get("cv_speech_only_penalty_applied"):
            return "skill set leans heavily CV/speech without NLP/IR depth"
        return "few core retrieval/ranking/vector-DB skills with real endorsement or duration backing"

    if key == "experience":
        if yoe is not None and yoe < 5.0:
            return f"only {yoe:.1f} years of experience, below the JD's 5-9 year band"
        if yoe is not None and yoe > 9.0:
            return f"{yoe:.1f} years of experience is above the JD's target band"
        return "experience-level fit is mixed"

    if key == "location":
        bucket = ev.get("location_bucket")
        loc = ev.get("location")
        if bucket == "outside_india_no_relocate":
            return f"based in {loc} with no relocation flag set"
        if bucket == "other_india":
            return f"based in {loc}, outside the JD's preferred Pune/Noida/NCR cities"
        return f"location ({loc}) is a soft mismatch"

    if key == "education":
        if ev.get("is_consulting_firm"):
            return f"currently at {ev.get('current_company')}, a services/consulting firm the JD is wary of"
        return "education tier is below average"

    return "this aspect is a weaker point"


def _behavioral_clause(feat: CandidateFeatures) -> str | None:
    ev = feat.evidence
    last_active = ev.get("last_active_days_ago")
    resp = ev.get("recruiter_response_rate")
    open_to_work = ev.get("open_to_work_flag")

    if last_active is not None and last_active > 120:
        return f"though they've been inactive on the platform for ~{last_active} days, dampening availability"
    if resp is not None and resp < 0.3:
        return f"recruiter response rate is low ({resp:.0%}), a soft availability concern"
    if open_to_work and last_active is not None and last_active <= 14 and resp is not None and resp >= 0.6:
        return f"and they're active, open-to-work, with a strong {resp:.0%} recruiter response rate"
    return None


def generate_reasoning(feat: CandidateFeatures, rng: random.Random) -> str:
    scores = _component_scores(feat)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    strongest_key, strongest_val = ranked[0]
    weakest_key, weakest_val = ranked[-1]

    ev = feat.evidence
    title = ev.get("current_title", "Candidate")
    yoe = ev.get("years_of_experience", 0.0)

    opener_templates = [
        f"{title} with {yoe:.1f} years of experience: ",
        f"{yoe:.1f}-year {title}: ",
        f"With {yoe:.1f} years as a {title}, ",
    ]
    opener = rng.choice(opener_templates)

    strength = _strength_phrase(feat, strongest_key)

    if feat.is_honeypot:
        flags = ", ".join(feat.evidence.get("honeypot_flags", []))
        return (
            f"{opener}profile shows internal inconsistencies ({flags}) that suggest "
            f"an implausible/fabricated history; scored near zero and excluded from "
            f"serious consideration despite surface-level keyword matches."
        )

    behav = _behavioral_clause(feat)

    # High-fit candidates: lead with title strength, then add a second,
    # different piece of evidence (skills detail, experience, location,
    # or behavioral) so reasonings vary even across many Tier-A titles.
    if feat.final_score >= 0.55:
        sentence = f"{opener}{strength}"

        # Pick a secondary detail that is NOT the same component as the
        # primary strength, preferring concrete skill names / yoe / location.
        secondary_candidates = [k for k, _ in ranked[1:] if k != strongest_key]
        added_secondary = False
        for key in secondary_candidates:
            if key == "skills":
                skills_phrase = _format_skills_phrase(feat)
                if skills_phrase:
                    sentence += f", with hands-on experience in {skills_phrase}"
                    added_secondary = True
                    break
            elif key == "experience" and strongest_key != "experience":
                sentence += f", and {yoe:.1f} years of experience sits within the JD's target band"
                added_secondary = True
                break
            elif key == "location" and strongest_key != "location":
                sentence += f", based in {ev.get('location')}"
                added_secondary = True
                break

        if weakest_val < 0.5 and weakest_key != strongest_key:
            sentence += f"; concern: {_concern_phrase(feat, weakest_key)}"
        if behav:
            sentence += f"; {behav}"
        sentence += "."
        return sentence

    # Mid-fit: balanced strength + concern
    if feat.final_score >= 0.25:
        sentence = (
            f"{opener}{strength}, but {_concern_phrase(feat, weakest_key)}"
        )
        if behav:
            sentence += f"; {behav}"
        sentence += "."
        return sentence

    # Low-fit: lead with the dominant concern, acknowledge any redeeming factor briefly
    sentence = f"{opener}{_concern_phrase(feat, weakest_key)}"
    if strongest_val >= 0.5:
        sentence += f"; on the positive side, {strength}"
    if behav:
        sentence += f"; {behav}"
    sentence += "."
    return sentence
