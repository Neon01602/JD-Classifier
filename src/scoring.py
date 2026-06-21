"""
Composite scoring: combine component scores from features.py into a single
final_score in [0, 1], with the honeypot hard-cap.

Weights (additive components, sum to 1.0 before behavioral multiplier):
    title:       0.35
    skills:      0.20
    experience:  0.15
    location:    0.10
    education:   0.05
  (sum = 0.85; behavioral_multiplier applies on top, range ~[0.4, 1.1],
   so effective max ~0.85*1.1 ≈ 0.935 plus headroom; we additionally fold
   in 0.15 reserved weight as part of the multiplier's effective range by
   rescaling — see compute_final_score for the exact formula.)

A small tiebreak signal (skill assessment scores, profile completeness,
recruiter interest) is blended in at low weight purely to spread out
candidates that would otherwise land on identical rounded scores --
this increases ranking granularity (helping NDCG) without changing the
ordinal ranking implied by the main components.

Title gate
----------
The 0.35 weight alone does not make title "decisive" in an additive
formula: a candidate with a near-zero title_score can still claw back
most of the lost ground via experience + location + education + a
generous behavioral multiplier. Empirically (see analysis on the sample
candidates), this let off-target-title candidates (tier E, e.g. "HR
Manager") land in the top quartile of a 50-candidate sample purely on
strong experience fit -- exactly the failure mode the JD calls out
("a candidate who has all the AI keywords ... but whose title is
Marketing Manager is not a fit, no matter how perfect their skill list
looks").

GATE_TABLE below applies a multiplicative ceiling keyed off the
*continuous* title_score (not the categorical tier letter), so the
career-history "rescue" path in features.score_title -- which already
boosts title_score for buzzword-free Tier 5s with real production
retrieval/ranking work -- is respected rather than punished. Only
candidates whose title_score stays low *after* the rescue check is
applied get capped.
"""

from __future__ import annotations

from features import CandidateFeatures

# Additive weights for the "static fit" components.
W_TITLE = 0.35
W_SKILLS = 0.20
W_EXPERIENCE = 0.15
W_LOCATION = 0.10
W_EDUCATION = 0.05
# Sum = 0.85. The remaining headroom (0.15) is realized through the
# behavioral multiplier, which scales the static-fit subtotal.

# Weight of the continuous tiebreak signal, scaled small enough that it can
# only shift the score by up to ~0.01 -- enough to separate ties at the
# 4-decimal display precision without reordering across meaningfully
# different static-fit + behavioral scores.
W_TIEBREAK = 0.01

HONEYPOT_CAP = 0.05

# (title_score threshold, gate multiplier) pairs, checked low-to-high.
# A title_score below 0.35 means: no rescue fired AND the base tier is D/E
# (see taxonomy.TITLE_TIER_SCORES) -- i.e. genuinely off-target. 0.35-0.60
# covers Tier C and "soft rescue" cases. 0.60+ is Tier A/B or a strong rescue.
GATE_TABLE = (
    (0.35, 0.40),
    (0.60, 0.75),
    (1.01, 1.00),
)


def title_gate(title_score: float) -> float:
    for threshold, gate in GATE_TABLE:
        if title_score < threshold:
            return gate
    return 1.00  # unreachable given the 1.01 sentinel, kept for clarity


def compute_final_score(feat: CandidateFeatures) -> float:
    static_fit = (
        W_TITLE * feat.title_score
        + W_SKILLS * feat.skills_score
        + W_EXPERIENCE * feat.experience_score
        + W_LOCATION * feat.location_score
        + W_EDUCATION * feat.education_score
    )

    gate = title_gate(feat.title_score)
    score = static_fit * gate * feat.behavioral_multiplier
    score += W_TIEBREAK * feat.tiebreak_score

    if feat.is_honeypot:
        score = min(score, HONEYPOT_CAP)

    feat.final_score = round(min(1.0, max(0.0, score)), 6)
    feat.evidence["title_gate_applied"] = gate
    return feat.final_score