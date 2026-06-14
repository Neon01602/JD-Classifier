
# Redrob Hackathon: JD-Classifier (Intelligent Candidate Ranking)
## From Dataset to Solution — A Complete Thinking Guide

---

## 1. The Problem

Redrob is an AI-powered hiring platform. The hackathon challenge was simple to state but hard to solve well:

> **Given a pool of 100,000 candidate profiles, find and rank the top 100 best fits for a "Senior AI Engineer – Founding Team" role.**

The output required: a CSV with `candidate_id`, `rank`, `score`, and `reasoning` for the top 100 candidates.

The deceptively hard part: the job description (JD) itself warned that a naive keyword-matching approach would fail, and explained exactly how and why.

---

## 2. Understanding the Dataset

### 2.1 What a Candidate Record Contains

Each candidate is one JSON object with eight top-level sections:

| Section | What It Contains | Why It Matters |
|---|---|---|
| `profile` | Title, company, industry, location, years of experience, headline, summary | First-pass signal: is this person even in the right domain? |
| `skills` | List of skill objects, each with `name`, `proficiency`, `duration_months`, `endorsements` | The richest structured signal — but also the most gameable |
| `career_history` | Array of past roles: title, company, industry, duration, description text | The ground truth of what someone actually did vs. what they claim |
| `education` | Degrees, institutions, graduation years, tier | Secondary signal for early-career candidates |
| `certifications` | Certification names and issuers | Weak signal — self-reported |
| `languages` | Language proficiency | Mostly irrelevant for this role |
| `redrob_signals` | Platform behavioral data: last active date, response rate, notice period, GitHub score, interview completion, verification flags | Availability and hireability — often ignored by naive rankers |
| `candidate_id` | Unique identifier (`CAND_XXXXXXX`) | Output key |

### 2.2 Key Dataset Characteristics

- **Scale**: 100,000 candidates — too large for LLM-based scoring per candidate without a GPU budget.
- **Skill diversity**: 133+ distinct skill names appear across the pool, ranging from core ML/IR skills (Embeddings, FAISS, RAG) to completely unrelated ones (Photoshop, Sales, Content Writing, Tailwind CSS).
- **Title diversity**: Candidates include HR Managers, Civil Engineers, Accountants, Graphic Designers — not just engineers. The dataset is deliberately noisy.
- **Behavioral signals**: Every candidate has platform engagement data — when they last logged in, how often they reply to recruiters, whether their phone/email is verified, their GitHub activity score.
- **Skill proficiency is self-reported**: A candidate can claim "expert" in any skill with zero months of usage and zero endorsements. This is a data quality problem the ranker must handle.

### 2.3 The Core Data Quality Problem

The `skills` field is a structured list — easy to query — but it is **entirely self-reported**. A candidate can list every buzzword from the JD: RAG, Pinecone, LLMs, FAISS, Embeddings. This is called **keyword-stuffing**, and it's the most common failure mode for resume parsers and naive rankers alike.

The `career_history[].description` field, by contrast, is free text describing what the person actually did in each role. It is harder to fake, because it requires constructing a plausible narrative, and it contains language patterns that betray real production experience vs. theoretical exposure.

This asymmetry — structured but gameable skills vs. unstructured but honest career text — is central to the entire design philosophy of the ranker.

---

## 3. Understanding the Job Description

The JD was for a **Senior AI Engineer on the Founding Team** at Redrob. Key requirements:

### What the JD explicitly asked for:
- 6–9 years of experience, mostly in applied ML at **product companies** (not IT services/consulting)
- Shipped a **production** ranking, search, retrieval, or recommendation system at meaningful scale
- Hands-on with **vector databases** (Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch)
- Experience with **hybrid search** (dense + sparse), embedding drift, index refresh
- Strong **Python** and software engineering discipline
- Designed **offline and online evaluation frameworks** (NDCG, MRR, A/B testing)
- Familiarity with **LLMs, RAG, fine-tuning** (LoRA, QLoRA, PEFT)
- Open source contributions, blog posts, or published work

### What the JD explicitly warned against:
1. **Keyword-stuffers** — candidates who list RAG/Pinecone/LLMs but whose career history shows no production search or retrieval work
2. **Pure researchers** — academics with publications but no deployed systems
3. **Consulting-only backgrounds** — TCS/Infosys/Wipro/Accenture career paths with no product company experience
4. **CV/Speech/Robotics specialists** — deep expertise in image classification or ASR but zero IR or LLM exposure
5. **Recent AI dabblers** — people who picked up LangChain and OpenAI in the last 12 months with no prior ML depth
6. **Ghost candidates** — perfect on paper but inactive, unresponsive, or unavailable

These explicit warnings shaped the ranker's disqualifier logic directly.

---

## 4. Why Naive Approaches Fail

### Approach 1: Keyword Match on Skills
Count how many JD keywords appear in the skills list. Fast, simple.

**Why it fails**: A candidate who just discovered LangChain last month can list every AI keyword in the JD. A seasoned search engineer who built Elasticsearch clusters for 5 years might not have "Vector Search" as a listed skill because the term didn't exist when they learned it. Keyword match rewards self-promotion, not expertise.

### Approach 2: Embedding / Semantic Search Only
Embed the JD and all candidate profiles, return top cosine similarities. Captures meaning better than keywords.

**Why it fails**: Semantic similarity doesn't distinguish between "I deployed this in production serving 2M users" and "I read a paper about this and built a prototype." It also can't apply business rules like "penalise consulting-only backgrounds" or "flag ghost candidates." And at 100K candidates with a CPU-only budget, naively calling a hosted embedding API per candidate is too slow and expensive.

### Approach 3: LLM-per-candidate scoring
Send each candidate to GPT/Claude and ask for a score.

**Why it fails**: 100,000 API calls × ~2,000 tokens each = 200M tokens. Cost-prohibitive and far too slow. No GPU, no hosted model budget was available per the submission spec.

---

## 5. The Solution Architecture

The ranker uses **six independent signal groups** combined into a composite score, then adjusted by **two multiplier layers** (disqualifiers and coherence checks).

```
Final Score = Composite × Availability Factor × Disqualifier Multiplier × Coherence Multiplier
```

### 5.1 The Six Signal Groups

#### Signal 1 — Skill Score (weight: 30%)
Not a raw keyword count. Each skill is evaluated across five **domain categories**, and each skill's contribution is weighted by:

- **Proficiency level** (beginner = 0.25 → expert = 1.0)
- **Trust factor** = a function of `duration_months` and `endorsements`
  - A skill claimed at "expert" with 0 months and 0 endorsements contributes nearly zero
  - A skill used for 24+ months with 20+ endorsements contributes full weight
  - Formula: `trust = (duration/24) × (0.5 + 0.5 × endorsements/20)`

The five categories and their weights within skill score:

| Category | Skills | Weight | Why |
|---|---|---|---|
| Core IR/Ranking | Vector Search, Embeddings, BM25, Semantic Search, Learning to Rank, etc. | 35% | The core of the role |
| Vector DBs | Pinecone, FAISS, Weaviate, Qdrant, Milvus, Elasticsearch, OpenSearch | 20% | Production retrieval stack |
| LLM/GenAI | LLMs, RAG, LangChain, Fine-tuning, LoRA, PEFT | 20% | Modern AI layer |
| Python/ML Core | Python, PyTorch, scikit-learn, NLP, MLOps, Deep Learning | 15% | Foundation skills |
| Infra/Scale | Kubernetes, Kafka, Spark, Docker, Airflow, Redis | 10% | Scale and deployment |

An **evaluation framework bonus** (up to +15%) is added for candidates whose career text mentions NDCG, MRR, A/B testing, precision@k — signals that they have actually measured and improved ranking systems, not just built them.

#### Signal 2 — Role/Title Fit (weight: 15%)
Current title and the three most recent titles are mapped to three tiers:

- **High fit** (1.0): AI Engineer, ML Engineer, NLP Engineer, Search Engineer, Applied Scientist, Data Scientist, Recommendation Systems Engineer
- **Medium fit** (0.45): Software Engineer, Backend Engineer, Data Engineer, Analytics Engineer
- **Low fit** (0.1): Everything else (HR Manager, Graphic Designer, Accountant, etc.)

Score = 70% current title + 30% best recent title. A seniority modifier (+0.10) applies if "Senior", "Staff", "Lead", or "Principal" appears in the current title.

**Why this matters**: The JD is for a senior IC role. A Marketing Manager with great AI skills is a harder sell than an ML Engineer with equivalent skills.

#### Signal 3 — Production Depth (weight: 20%)
This is where career history text is parsed for language patterns that signal real production experience vs. theoretical exposure.

**Production indicators** (regex-matched): `production`, `deployed`, `shipped`, `scale`, `real users`, `live system`, `on-call`, `monitoring`, `latency`, `throughput`, `rollout`, `a/b test`, `millions of users`

**Research-only indicators**: `research`, `paper`, `arxiv`, `academic`, `experiment`, `prototype`, `proof of concept`, `poc`

Earlier roles are down-weighted: current role gets 1.5×, previous role 1.0×, older roles 0.5×. This prevents a candidate who did production work 8 years ago and has since moved to pure research from scoring high.

**Why production depth gets 20%**: The JD explicitly says "shipped" systems serving "real users." A great researcher who has never deployed anything is not the right hire for a founding team role.

#### Signal 4 — Semantic Similarity (weight: 15%)
TF-IDF cosine similarity between each candidate's full text representation and a hand-crafted "ideal candidate" document. The ideal document is a dense, first-person narrative paraphrase of the JD's "things you absolutely need" and "how to read between the lines" sections.

This is the only signal that captures nuance in career narrative language that structured signals miss — words like "embedding drift", "retrieval quality regression", "offline to online correlation."

All 100,000 candidates are vectorised in a single `sklearn.TfidfVectorizer.fit_transform()` call, making it fast enough to run on CPU in under a minute even at full scale. The ideal document is included in the fit corpus so its vocabulary is always represented.

#### Signal 5 — Company/Industry Fit (weight: 10%)
Based on the current employer's industry:

- **Product industries** (AI/ML, SaaS, Fintech, EdTech, E-commerce, etc.) → 1.0
- **IT Services / Consulting** with prior product company experience → 0.75
- **IT Services / Consulting** with no product background → 0.4
- **Other industries** → 0.6

**Why this matters**: The JD explicitly says "most of it in applied ML at product companies, not pure IT services or consulting." Someone who has spent their entire career at TCS or Infosys, even if technically skilled, has developed in a very different environment than a fast-moving AI product company.

#### Signal 6 — Location Fit (weight: 10%)
The role is based in Pune/Noida. Score tiers:

- Pune or Noida → 1.0 (already there)
- Tier-1 India cities (Bangalore, Mumbai, Delhi, Gurgaon, Hyderabad, Chennai) → 0.9 (easy relocation)
- Rest of India → 0.75
- International, willing to relocate → 0.5
- International, not willing → 0.2

### 5.2 Availability as a Multiplier (not a signal)

Availability is **not added to the composite** — it **multiplies it**, mapped to `[0.4, 1.0]`. This prevents a ghost candidate with a perfect technical profile from ranking in the top 100.

The availability score is built from:
- Base: 0.55 (everyone starts here — being on the platform at all is a signal)
- Open to work flag: +0.10
- Recency (days since last active, decays over 180 days): up to +0.15
- Recruiter response rate: up to +0.15
- Notice period (shorter is better, scales over 90 days): up to +0.10
- Interview completion rate: up to +0.05
- Verification (email + phone + LinkedIn): up to +0.05

### 5.3 Disqualifier Multipliers

Six red-flag conditions from the JD map to score multipliers:

| Flag | Condition | Multiplier | Reasoning |
|---|---|---|---|
| `research_only` | Zero production hits in career text + ≥2 research hits + weak IR skills | ×0.15 | The role requires shipped systems, not papers |
| `consulting_only` | Entire career at TCS/Infosys/Wipro/Accenture/etc. | ×0.20 | Product company experience explicitly required |
| `cv_speech_robotics_only` | 2+ CV/Speech/Robotics skills, zero IR/LLM skills | ×0.30 | Different domain; skill transfer is not given |
| `recent_ai_only` | GenAI skills all <12 months old, no deep ML background ≥24 months | ×0.50 | Dabbler, not a practitioner |
| `title_chaser` | 2+ short stints (<18 months) at different companies with escalating seniority titles | ×0.60 | Job-hopping for titles, not for learning |
| `closed_source_no_validation` | 5+ YOE, GitHub score = -1, no open-source ML | ×0.90 | Mild penalty — JD values external validation |

Multiple flags stack multiplicatively. A research-only consulting-only candidate would score ×0.03 — effectively eliminated.

### 5.4 Coherence Multipliers (Honeypot Detection)

Three data integrity checks catch profiles that are internally inconsistent — either synthetic, inflated, or corrupted:

| Flag | Condition | Multiplier |
|---|---|---|
| `career_duration_mismatch` | Sum of career history months < 50% or > 180% of stated years of experience | ×0.05 |
| `expert_with_zero_tenure` | Any skill claimed as "expert" with 0 months of use | ×0.05 |
| `skill_assessment_mismatch` | Redrob's skill assessment scores reference a skill not in the skills list | ×0.50 |

The extreme penalties (×0.05) for the first two are intentional — these are near-certain signs of a fabricated or badly corrupted profile.

---

## 6. The Final Scoring Formula

```
composite = (
    0.30 × skill_score
  + 0.15 × role_fit_score
  + 0.20 × production_depth_score
  + 0.15 × semantic_similarity_score
  + 0.10 × company_fit_score
  + 0.10 × location_score
)

availability_factor = 0.4 + 0.6 × availability_score   # maps [0,1] → [0.4, 1.0]

final_score = composite × availability_factor × disqualifier_mult × coherence_mult
```

All scores are clamped to `[0.0, 1.0]`.

---

## 7. The Output Format

The CSV has four columns matching the sample submission exactly:

```
candidate_id, rank, score, reasoning
CAND_0000031, 1, 0.8349, Recommendation Systems Engineer with 6.0 yrs; 12 AI core skills; response rate 0.91.
```

The **reasoning string** format: `"{title} with {yoe} yrs; {n} AI core skills; response rate {rate}."`

- `title` = current job title from profile
- `yoe` = years of experience (1 decimal place)
- `n` = count of skills that belong to any of the five AI skill categories
- `rate` = recruiter response rate from redrob_signals (2 decimal places)

This is intentionally human-readable — a recruiter or hiring manager can scan the top 100 and immediately understand why each candidate ranked where they did.

---

## 8. Engineering Decisions

### Why TF-IDF instead of dense embeddings?
At 100K candidates with no GPU and a 5-minute time budget, dense embeddings via a hosted API (OpenAI, Cohere) would cost ~$50–200 and take 10–30 minutes. Sentence Transformers locally would need a GPU or ~20 minutes on CPU. TF-IDF with `sklearn` fits and transforms 100K documents in under 60 seconds on a single CPU core and captures enough vocabulary overlap to be a useful signal. It is the right tradeoff for this constraint.

### Why streaming (JSONL line-by-line)?
100K candidates at ~3KB each = ~300MB. Loading everything into memory at once would use ~600MB–1GB after Python object overhead. Streaming keeps memory bounded at a few MB for the in-progress batch and the growing `records` list (which stores only computed scores, not raw documents).

### Why multipliers instead of weights for availability and disqualifiers?
If availability were an additive component (say 10% weight), a ghost candidate with a 0.0 availability score would lose only 10 points off their composite — they could still rank in the top 100 on skill strength alone. As a multiplier, a candidate with near-zero availability drags their entire score toward zero regardless of skills. The JD's core insight is: a great candidate who won't respond is not actually hireable.

### Why not just rank on AI skill count?
The sample submission's reasoning string shows AI skill count — but it's only the *explanation*, not the score. The actual ranking scores are computed from the full six-signal composite. If you ranked purely on AI skill count, you'd surface candidates who keyword-stuffed their profile and ignore production depth, availability, location, and all the disqualifiers. The score reflects substance; the reasoning string provides transparency.

---

## 9. What the Top Candidates Look Like

Based on the 50-candidate sample, the correct top candidate (CAND_0000031) is a **Recommendation Systems Engineer** with:
- 6 years of experience
- 12 AI core skills (highest in sample) including direct IR and retrieval skills
- 0.91 recruiter response rate (most responsive in sample)
- Active at a relevant company

The ranker gives them a score of **0.8349**, comfortably ahead of the next candidate at 0.3423 — not because of a single signal, but because they score well across all six dimensions and trigger none of the disqualifiers.

By contrast, candidates with high AI skill counts but poor production depth, consulting-only backgrounds, or low availability end up far lower in the rankings despite looking impressive on a keyword scan.

---

## 10. Summary of Design Principles

| Principle | Implementation |
|---|---|
| Don't trust self-reported skills at face value | Trust factor: endorsements × duration weighting |
| Career text reveals more than skills list | Production/research pattern matching on job descriptions |
| "Available on paper" ≠ hireable | Availability as a score multiplier, not additive weight |
| Business rules matter as much as ML | Explicit disqualifiers for consulting-only, research-only, recent-dabbler profiles |
| Data integrity must be verified | Coherence checks for mismatched career timelines and impossible skill claims |
| Must run on CPU in under 5 minutes | TF-IDF over dense embeddings; streaming JSONL; single-pass vectorisation |
| Reasoning must be human-readable | Standardised reasoning string format matching sample submission |
