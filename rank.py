#!/usr/bin/env python3
"""
rank.py - Redrob Hackathon: Intelligent Candidate Discovery & Ranking
=======================================================================

Produces a top-100 ranked CSV (candidate_id, rank, score, reasoning) from the
100K-candidate pool, scored against the "Senior AI Engineer - Founding Team"
job description.

USAGE
-----
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

candidates.jsonl(.gz) is read line-by-line (streaming).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# 0. CONSTANTS
# ============================================================================

REFERENCE_DATE = date(2026, 6, 1)
REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
TOP_K = 100

# ============================================================================
# 1. SKILL TAXONOMY
# ============================================================================

CORE_IR_RANKING = {
    "Information Retrieval", "Information Retrieval Systems", "BM25",
    "Vector Search", "Embeddings", "Sentence Transformers", "Semantic Search",
    "Recommendation Systems", "Learning to Rank", "Ranking Systems",
    "Search Backend", "Search Infrastructure", "Search & Discovery",
    "Indexing Algorithms", "Vector Representations", "Text Encoders",
    "Content Matching",
}

VECTOR_DB_HYBRID_SEARCH = {
    "Pinecone", "FAISS", "Weaviate", "Qdrant", "Milvus",
    "Elasticsearch", "OpenSearch", "pgvector",
}

LLM_GENAI = {
    "LLMs", "RAG", "LangChain", "LlamaIndex", "Haystack",
    "Hugging Face Transformers", "Prompt Engineering", "Fine-tuning LLMs",
    "LoRA", "QLoRA", "PEFT", "Model Adaptation",
}

PYTHON_ML_CORE = {
    "Python", "PyTorch", "TensorFlow", "scikit-learn", "Machine Learning",
    "Deep Learning", "NLP", "Natural Language Processing",
    "Feature Engineering", "MLOps", "Open-source ML libraries", "Data Science",
}

INFRA_SCALE = {
    "Kubernetes", "Spark", "Kafka", "Docker", "Airflow",
    "Workflow Orchestration", "Microservices", "MLflow", "Kubeflow",
    "BentoML", "Document Processing", "Apache Beam", "Apache Flink",
    "Hadoop", "Redis", "Databricks",
}

CV_SPEECH_ROBOTICS = {
    "Computer Vision", "Image Classification", "Object Detection", "OpenCV",
    "CNN", "GANs", "Diffusion Models", "YOLO", "Speech Recognition", "TTS",
    "ASR", "Reinforcement Learning",
}

# All AI-relevant skills (used for reasoning string count)
ALL_AI_SKILLS = (
    CORE_IR_RANKING | VECTOR_DB_HYBRID_SEARCH | LLM_GENAI | PYTHON_ML_CORE | INFRA_SCALE
)

SKILL_CATEGORIES = [
    ("core_ir_ranking", CORE_IR_RANKING, 0.35, 3.0),
    ("vector_db",       VECTOR_DB_HYBRID_SEARCH, 0.20, 2.0),
    ("llm_genai",       LLM_GENAI,  0.20, 2.5),
    ("python_ml_core",  PYTHON_ML_CORE, 0.15, 2.5),
    ("infra_scale",     INFRA_SCALE, 0.10, 2.5),
]

PROFICIENCY_WEIGHT = {
    "beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0,
}

# ============================================================================
# 2. TEXT PATTERNS
# ============================================================================

PRODUCTION_PATTERNS = re.compile(
    r"\b(production|deployed|deployment|shipped?|shipping|scale|scaled|"
    r"scaling|scalable|real users?|live system|on-call|monitoring|latency|"
    r"throughput|rollout|a/?b test\w*|millions of (users|items|requests)|"
    r"end-to-end)\b",
    re.IGNORECASE,
)

RESEARCH_ONLY_PATTERNS = re.compile(
    r"\b(research(-only)?|publications?|paper|arxiv|academic|"
    r"experiments?|prototypes?|proof of concept|\bpoc\b)\b",
    re.IGNORECASE,
)

EVAL_FRAMEWORK_PATTERNS = re.compile(
    r"\b(ndcg|mrr|\bmap\b|mean average precision|precision@|recall@|"
    r"a/?b test\w*|offline evaluation|online evaluation|"
    r"evaluation framework|ranking metric)\b",
    re.IGNORECASE,
)

SENIORITY_TITLE_PATTERN = re.compile(
    r"\b(senior|staff|lead|principal|director|head)\b", re.IGNORECASE
)

# ============================================================================
# 3. ROLE / TITLE FIT
# ============================================================================

HIGH_FIT_TITLES = {
    "ai engineer", "senior ai engineer", "lead ai engineer", "staff ai engineer",
    "machine learning engineer", "senior machine learning engineer",
    "staff machine learning engineer", "applied ml engineer",
    "nlp engineer", "senior nlp engineer",
    "search engineer", "recommendation systems engineer",
    "applied scientist", "senior applied scientist",
    "data scientist", "senior data scientist",
    "ai research engineer", "junior ml engineer",
}

MEDIUM_FIT_TITLES = {
    "software engineer", "senior software engineer", "staff software engineer",
    "backend engineer", "full stack developer",
    "data engineer", "senior data engineer", "analytics engineer",
    "data analyst",
}

TITLE_FIT_WEIGHT = {"high": 1.0, "medium": 0.45, "low": 0.1}


def title_fit_tier(title: str) -> str:
    t = title.lower().strip()
    if t in HIGH_FIT_TITLES:
        return "high"
    if t in MEDIUM_FIT_TITLES:
        return "medium"
    return "low"


# ============================================================================
# 4. COMPANY / INDUSTRY FIT
# ============================================================================

PRODUCT_INDUSTRIES = {
    "AI/ML", "AI Services", "Conversational AI", "Voice AI", "HealthTech AI",
    "SaaS", "Software", "Fintech", "E-commerce", "EdTech", "Internet",
    "Gaming", "HealthTech", "Insurance Tech", "AdTech",
}

SERVICES_INDUSTRIES = {"IT Services", "Consulting"}

CONSULTING_FIRMS = {
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
    "HCL", "Mindtree",
}

# ============================================================================
# 5. LOCATION FIT
# ============================================================================

PUNE_NOIDA = {"Pune, Maharashtra", "Noida, Uttar Pradesh"}

TIER1_INDIA = {
    "Bangalore, Karnataka", "Mumbai, Maharashtra", "Delhi, Delhi",
    "Gurgaon, Haryana", "Hyderabad, Telangana", "Chennai, Tamil Nadu",
}

# ============================================================================
# 6. COMPOSITE WEIGHTS
# ============================================================================

COMPOSITE_WEIGHTS = {
    "skill":            0.30,
    "role_fit":         0.15,
    "production_depth": 0.20,
    "semantic":         0.15,
    "company_fit":      0.10,
    "location":         0.10,
}
assert abs(sum(COMPOSITE_WEIGHTS.values()) - 1.0) < 1e-9

# ============================================================================
# 7. IDEAL CANDIDATE TEXT (TF-IDF anchor)
# ============================================================================

IDEAL_CANDIDATE_TEXT = """
Senior AI engineer with six to nine years of experience, most of it in
applied machine learning and AI roles at product companies, not pure IT
services or consulting. Has shipped an end-to-end ranking, search,
retrieval, or recommendation system that real users depend on at meaningful
scale. Comfortable across the stack from data infrastructure to ranking
algorithms to product judgment. Has hands-on production experience with
embeddings based retrieval, sentence transformers, dense and sparse vector
search, hybrid search, and vector databases such as pinecone, weaviate,
qdrant, milvus, faiss, elasticsearch, or opensearch deployed to real users,
including handling embedding drift, index refresh, and retrieval quality
regressions. Strong python and software engineering practices, code quality
matters. Hands on experience designing offline and online evaluation
frameworks for ranking systems including ndcg, mrr, mean average precision,
precision at k, and a/b testing, with offline to online correlation.
Experience with large language models, retrieval augmented generation,
prompt engineering, and fine-tuning with lora, qlora, or peft is valued.
Experience with learning to rank models such as xgboost based or neural
rankers is valued. Background in distributed systems, kafka, spark,
kubernetes, or large scale inference optimization is valued. Open source
contributions, blog posts, or published work that show how this person
thinks about systems are valued. Prefers shipping a working version one in
weeks over a perfect version two over months, and thinks about
recruiter-facing search, candidate-facing matching, hybrid retrieval, and
LLM based re-ranking as one connected intelligence layer.
""".strip()


# ============================================================================
# 8. DATA LOADING
# ============================================================================

def iter_candidates(path: Path):
    """Stream candidates.jsonl or candidates.jsonl.gz line by line.
    Also supports a plain .json file that is a top-level list."""
    if str(path).endswith(".gz"):
        opener = gzip.open
        mode = "rt"
    elif str(path).endswith(".json"):
        # Could be a JSON array (sample file) — load whole file
        with open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            yield from data
        else:
            yield data
        return
    else:
        opener = open
        mode = "rt"

    with opener(path, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ============================================================================
# 9. FEATURE EXTRACTION
# ============================================================================

def skill_lookup(skills: list[dict]) -> dict[str, dict]:
    return {s["name"]: s for s in skills}


def trust_factor(duration_months: int, endorsements: int) -> float:
    dur = min(1.0, max(0, duration_months) / 24.0)
    end = min(1.0, max(0, endorsements) / 20.0)
    return dur * (0.5 + 0.5 * end)


def category_strength(skills_by_name: dict, category_skills: set) -> float:
    total = 0.0
    for name in category_skills:
        s = skills_by_name.get(name)
        if not s:
            continue
        pw = PROFICIENCY_WEIGHT.get(s["proficiency"], 0.25)
        total += pw * trust_factor(s.get("duration_months", 0), s.get("endorsements", 0))
    return total


def skill_score_and_breakdown(skills_by_name: dict, career_text: str) -> tuple[float, dict]:
    breakdown = {}
    score = 0.0
    for cat_name, cat_skills, weight, saturation in SKILL_CATEGORIES:
        strength = category_strength(skills_by_name, cat_skills)
        normed = min(1.0, strength / saturation) if saturation > 0 else 0.0
        breakdown[cat_name] = normed
        score += weight * normed

    eval_hits = len(EVAL_FRAMEWORK_PATTERNS.findall(career_text))
    eval_bonus = min(0.15, 0.05 * eval_hits)
    breakdown["eval_bonus"] = eval_bonus
    score = min(1.0, score + eval_bonus)
    return score, breakdown


def role_fit_score_and_tier(profile: dict, career_history: list[dict]) -> tuple[float, str]:
    cur_title = profile["current_title"]
    cur_tier = title_fit_tier(cur_title)
    cur_w = TITLE_FIT_WEIGHT[cur_tier]

    recent_titles = [e["title"] for e in career_history[:3]]
    recent_w = max((TITLE_FIT_WEIGHT[title_fit_tier(t)] for t in recent_titles), default=0.0)

    score = 0.7 * cur_w + 0.3 * recent_w
    if SENIORITY_TITLE_PATTERN.search(cur_title):
        score = min(1.0, score + 0.1)
    return score, cur_tier


def production_research_signals(career_history: list[dict]) -> tuple[float, float, float]:
    weighted_hits = 0.0
    prod_total = 0
    research_total = 0
    weights = [1.5, 1.0] + [0.5] * 10
    for i, e in enumerate(career_history):
        w = weights[i] if i < len(weights) else 0.25
        desc = e.get("description", "")
        prod_hits = len(PRODUCTION_PATTERNS.findall(desc))
        research_hits = len(RESEARCH_ONLY_PATTERNS.findall(desc))
        prod_total += prod_hits
        research_total += research_hits
        weighted_hits += w * prod_hits
    depth = min(1.0, weighted_hits / 6.0)
    return depth, float(prod_total), float(research_total)


def company_fit_score(profile: dict, career_history: list[dict]) -> float:
    cur_industry = profile["current_industry"]
    if cur_industry in PRODUCT_INDUSTRIES:
        return 1.0
    if cur_industry in SERVICES_INDUSTRIES:
        prior_product = any(e["industry"] in PRODUCT_INDUSTRIES for e in career_history)
        return 0.75 if prior_product else 0.4
    return 0.6


def location_score(profile: dict, willing_to_relocate: bool) -> float:
    loc = profile.get("location", "")
    if loc in PUNE_NOIDA:
        return 1.0
    if loc in TIER1_INDIA:
        return 0.9
    if profile.get("country") == "India":
        return 0.75
    return 0.5 if willing_to_relocate else 0.2


def availability_score(rs: dict) -> float:
    last_active = date.fromisoformat(rs["last_active_date"])
    days_since = (REFERENCE_DATE - last_active).days
    recency = max(0.0, 1.0 - days_since / 180.0)

    notice = rs.get("notice_period_days", 90)
    notice_score = max(0.0, 1.0 - notice / 90.0)

    verification = (
        int(rs.get("verified_email", False))
        + int(rs.get("verified_phone", False))
        + int(rs.get("linkedin_connected", False))
    ) / 3.0

    score = 0.55
    score += 0.10 if rs.get("open_to_work_flag", False) else 0.0
    score += 0.15 * recency
    score += 0.15 * rs.get("recruiter_response_rate", 0.0)
    score += 0.10 * notice_score
    score += 0.05 * rs.get("interview_completion_rate", 0.0)
    score += 0.05 * verification
    return score


def disqualifier_multiplier(
    profile, career_history, skills_by_name, breakdown, prod_hits, research_hits, rs
) -> tuple[float, list[str]]:
    mult = 1.0
    flags = []

    all_companies = {profile["current_company"]} | {e["company"] for e in career_history}

    core_ir_strength = breakdown.get("core_ir_ranking", 0.0)
    if prod_hits == 0 and research_hits >= 2 and core_ir_strength < 0.3:
        mult *= 0.15
        flags.append("research_only")

    if all_companies <= CONSULTING_FIRMS:
        mult *= 0.20
        flags.append("consulting_only")

    cv_count = sum(1 for n in CV_SPEECH_ROBOTICS if n in skills_by_name)
    ir_llm_count = sum(
        1 for n in (CORE_IR_RANKING | LLM_GENAI | VECTOR_DB_HYBRID_SEARCH) if n in skills_by_name
    )
    if cv_count >= 2 and ir_llm_count == 0:
        mult *= 0.30
        flags.append("cv_speech_robotics_only")

    genai_durations = [
        skills_by_name[n].get("duration_months", 0)
        for n in (LLM_GENAI | VECTOR_DB_HYBRID_SEARCH)
        if n in skills_by_name
    ]
    deep_durations = [
        skills_by_name[n].get("duration_months", 0)
        for n in (CORE_IR_RANKING | PYTHON_ML_CORE)
        if n in skills_by_name
    ]
    has_genai_signal = (breakdown.get("llm_genai", 0) + breakdown.get("vector_db", 0)) > 0.3
    if has_genai_signal and genai_durations and max(genai_durations) < 12:
        if not deep_durations or max(deep_durations) < 24:
            mult *= 0.5
            flags.append("recent_ai_only")

    short_senior_stints = [
        e for e in career_history
        if e["duration_months"] < 18 and SENIORITY_TITLE_PATTERN.search(e["title"])
    ]
    if len({e["company"] for e in short_senior_stints}) >= 2:
        mult *= 0.6
        flags.append("title_chaser")

    if (
        profile["years_of_experience"] >= 5
        and rs.get("github_activity_score", -1) == -1
        and "Open-source ML libraries" not in skills_by_name
    ):
        mult *= 0.9
        flags.append("closed_source_no_validation")

    return mult, flags


def coherence_multiplier(profile, career_history, skills, rs) -> tuple[float, list[str]]:
    mult = 1.0
    flags = []

    yoe = profile["years_of_experience"]
    total_months = sum(e["duration_months"] for e in career_history)
    if yoe > 0:
        ratio = total_months / (yoe * 12.0)
        if ratio < 0.5 or ratio > 1.8:
            mult *= 0.05
            flags.append("career_duration_mismatch")

    for s in skills:
        if s["proficiency"] == "expert" and s.get("duration_months", 0) == 0:
            mult *= 0.05
            flags.append("expert_with_zero_tenure")
            break

    skill_names = {s["name"] for s in skills}
    for sk in rs.get("skill_assessment_scores", {}):
        if sk not in skill_names:
            mult *= 0.5
            flags.append("skill_assessment_mismatch")
            break

    return mult, flags


# ============================================================================
# 10. CANDIDATE TEXT (for TF-IDF)
# ============================================================================

def build_candidate_text(profile: dict, skills: list[dict], career_history: list[dict]) -> str:
    parts: list[str] = []
    parts.append(profile.get("current_title", ""))
    parts.append(profile.get("summary", ""))

    proficiency_repeats = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
    for s in skills:
        repeats = proficiency_repeats.get(s.get("proficiency", "beginner"), 1)
        parts.extend([s["name"]] * repeats)

    for e in career_history:
        parts.append(e.get("title", ""))
        parts.append(e.get("description", ""))

    return " ".join(p for p in parts if p)


# ============================================================================
# 11. REASONING STRING  ← matches sample_submission.csv format exactly
#     "{title} with {yoe} yrs; {n} AI core skills; response rate {rate}."
# ============================================================================

def build_reasoning(profile: dict, skills: list[dict], rs: dict) -> str:
    title = profile.get("current_title", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    skill_names = {s["name"] for s in skills}
    ai_skill_count = len(skill_names & ALL_AI_SKILLS)
    response_rate = rs.get("recruiter_response_rate", 0.0)
    return (
        f"{title} with {yoe:.1f} yrs; "
        f"{ai_skill_count} AI core skills; "
        f"response rate {response_rate:.2f}."
    )


# ============================================================================
# 12. MAIN PIPELINE
# ============================================================================

def run(candidates_path: Path, out_path: Path, verbose: bool = True) -> None:
    t0 = time.time()

    # ------------------------------------------------------------------
    # Pass 1: stream, compute non-semantic scores, collect corpus
    # ------------------------------------------------------------------
    if verbose:
        print(f"[rank] Pass 1: loading {candidates_path} …", flush=True)

    records: list[dict] = []
    corpus:  list[str]  = []

    n_loaded = 0
    for cand in iter_candidates(candidates_path):
        n_loaded += 1
        if verbose and n_loaded % 10_000 == 0:
            print(f"  … {n_loaded:,} candidates", flush=True)

        profile        = cand["profile"]
        skills         = cand.get("skills", [])
        career_history = cand.get("career_history", [])
        rs             = cand.get("redrob_signals", {})

        # willing_to_relocate lives inside redrob_signals in the real dataset
        willing = rs.get("willing_to_relocate", cand.get("willing_to_relocate", False))

        skills_by_name = skill_lookup(skills)
        career_text    = " ".join(e.get("description", "") for e in career_history)

        sk_sc, breakdown                  = skill_score_and_breakdown(skills_by_name, career_text)
        role_sc, role_tier                = role_fit_score_and_tier(profile, career_history)
        prod_sc, prod_hits, research_hits = production_research_signals(career_history)
        company_sc                        = company_fit_score(profile, career_history)
        loc_sc                            = location_score(profile, willing)
        avail_sc                          = availability_score(rs)

        dq_mult, dq_flags   = disqualifier_multiplier(
            profile, career_history, skills_by_name,
            breakdown, prod_hits, research_hits, rs
        )
        coh_mult, coh_flags = coherence_multiplier(profile, career_history, skills, rs)

        records.append({
            "candidate_id": cand["candidate_id"],
            "profile":      profile,
            "skills":       skills,
            "rs":           rs,
            "sk_sc":        sk_sc,
            "role_sc":      role_sc,
            "role_tier":    role_tier,
            "prod_sc":      prod_sc,
            "company_sc":   company_sc,
            "loc_sc":       loc_sc,
            "avail_sc":     avail_sc,
            "dq_mult":      dq_mult,
            "dq_flags":     dq_flags,
            "coh_mult":     coh_mult,
            "coh_flags":    coh_flags,
        })
        corpus.append(build_candidate_text(profile, skills, career_history))

    if verbose:
        print(f"[rank] Pass 1 done: {n_loaded:,} candidates in {time.time()-t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Pass 2: TF-IDF semantic similarity (batch)
    # ------------------------------------------------------------------
    if verbose:
        print("[rank] Pass 2: TF-IDF …", flush=True)

    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        min_df=2, max_df=0.95,
        sublinear_tf=True, max_features=60_000,
    )
    all_docs     = [IDEAL_CANDIDATE_TEXT] + corpus
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    ideal_vec    = tfidf_matrix[0]
    cand_matrix  = tfidf_matrix[1:]
    semantic_scores: np.ndarray = cosine_similarity(ideal_vec, cand_matrix).flatten()

    if verbose:
        print(f"[rank] TF-IDF done in {time.time()-t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Pass 3: composite score + multipliers → sort → top-K
    # ------------------------------------------------------------------
    W = COMPOSITE_WEIGHTS
    final_scores: list[tuple[float, int]] = []

    for idx, rec in enumerate(records):
        sem_sc = float(semantic_scores[idx])

        composite = (
            W["skill"]            * rec["sk_sc"]
            + W["role_fit"]       * rec["role_sc"]
            + W["production_depth"] * rec["prod_sc"]
            + W["semantic"]       * sem_sc
            + W["company_fit"]    * rec["company_sc"]
            + W["location"]       * rec["loc_sc"]
        )

        # Availability: mapped to [0.4, 1.0] so ghost candidates are penalised
        # without being zeroed out entirely
        avail_factor = 0.4 + 0.6 * rec["avail_sc"]
        score = composite * avail_factor * rec["dq_mult"] * rec["coh_mult"]
        score = max(0.0, min(1.0, score))

        rec["sem_sc"]     = sem_sc
        rec["final_score"] = score
        final_scores.append((score, idx))

    final_scores.sort(key=lambda x: x[0], reverse=True)
    top_k = final_scores[:TOP_K]

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    if verbose:
        print(f"[rank] Writing {out_path} …", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REQUIRED_HEADER)
        writer.writeheader()
        for rank, (score, idx) in enumerate(top_k, start=1):
            rec = records[idx]
            writer.writerow({
                "candidate_id": rec["candidate_id"],
                "rank":         rank,
                "score":        round(score, 4),
                "reasoning":    build_reasoning(rec["profile"], rec["skills"], rec["rs"]),
            })

    elapsed = time.time() - t0
    if verbose:
        print(f"[rank] Done — {n_loaded:,} candidates in {elapsed:.1f}s → {out_path}")
        if top_k:
            best = records[top_k[0][1]]
            print(f"[rank] #1: {best['candidate_id']}  score={top_k[0][0]:.4f}")


# ============================================================================
# 13. CLI
# ============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Redrob Hackathon — rank candidates for Senior AI Engineer role.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--candidates", type=Path, required=True,
                   help="candidates.jsonl / .jsonl.gz / .json")
    p.add_argument("--out", type=Path, default=Path("submission.csv"),
                   help="Output CSV path")
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(candidates_path=args.candidates, out_path=args.out, verbose=not args.quiet)
