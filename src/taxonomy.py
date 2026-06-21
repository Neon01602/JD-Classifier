"""
Static taxonomy and weight tables for the Redrob ranker.

These tables encode the JD's "what we actually need" reading:
  - Title-trajectory match is the decisive signal (JD final note).
  - Skills are evaluated against a JD-derived "core stack", with
    endorsement/duration as a trust multiplier against keyword-stuffing.
  - Pure-services-only career history and pure-research-only career
    history are explicit disqualifying patterns per the JD.
"""

# ---------------------------------------------------------------------------
# 1. Title tiers — how closely does current_title match the JD's target role?
#    Score is a base multiplier in [0, 1] applied to the title component.
# ---------------------------------------------------------------------------

# Tier A: titles that are essentially the role itself.
TITLE_TIER_A = {
    "senior ai engineer",
    "lead ai engineer",
    "senior machine learning engineer",
    "staff machine learning engineer",
    "senior nlp engineer",
    "senior applied scientist",
    "applied ml engineer",
}

# Tier B: strong adjacent roles — IR/ranking/search/recsys specialists,
# or general ML engineers.
TITLE_TIER_B = {
    "machine learning engineer",
    "ml engineer",
    "nlp engineer",
    "recommendation systems engineer",
    "search engineer",
    "ai engineer",
    "senior data scientist",
    "data scientist",
    "senior software engineer (ml)",
    "computer vision engineer",  # JD explicitly deprioritizes CV-only, handled separately
    "ai specialist",
    "ai research engineer",
}

# Tier C: junior versions of the above, or strong data/backend roles that
# plausibly ship ranking/retrieval-adjacent systems.
TITLE_TIER_C = {
    "junior ml engineer",
    "senior data engineer",
    "data engineer",
    "analytics engineer",
    "data analyst",
    "backend engineer",
    "senior software engineer",
    "software engineer",
}

# Tier D: generalist engineering roles — plausible but need strong
# career-history evidence of ML/retrieval work to score well.
TITLE_TIER_D = {
    "full stack developer",
    "cloud engineer",
    "devops engineer",
    "java developer",
    ".net developer",
    "mobile developer",
    "frontend engineer",
    "qa engineer",
}

# Tier E: clearly non-technical / unrelated current titles. These are the
# "9 AI keywords but title is HR Manager" trap candidates.
TITLE_TIER_E = {
    "business analyst",
    "hr manager",
    "mechanical engineer",
    "accountant",
    "project manager",
    "customer support",
    "operations manager",
    "content writer",
    "sales executive",
    "civil engineer",
    "graphic designer",
    "marketing manager",
}

TITLE_TIER_SCORES = {
    "A": 1.00,
    "B": 0.80,
    "C": 0.55,
    "D": 0.30,
    "E": 0.03,   # near-zero but not exactly zero; career history can still rescue
}


def title_tier_and_score(title: str):
    t = title.strip().lower()
    if t in TITLE_TIER_A:
        return "A", TITLE_TIER_SCORES["A"]
    if t in TITLE_TIER_B:
        return "B", TITLE_TIER_SCORES["B"]
    if t in TITLE_TIER_C:
        return "C", TITLE_TIER_SCORES["C"]
    if t in TITLE_TIER_D:
        return "D", TITLE_TIER_SCORES["D"]
    if t in TITLE_TIER_E:
        return "E", TITLE_TIER_SCORES["E"]
    # Unknown title -> treat as Tier D (neutral-low, let career history decide)
    return "D", TITLE_TIER_SCORES["D"]


# ---------------------------------------------------------------------------
# 2. JD core skill stack — "things you absolutely need"
#    Weighted so retrieval/embeddings/vector-DB/eval/Python dominate.
# ---------------------------------------------------------------------------

# Each skill maps to (category, weight). Weight is relative importance
# within the skills component.
CORE_SKILLS = {
    # Embeddings-based retrieval
    "embeddings": ("retrieval", 1.0),
    "vector representations": ("retrieval", 1.0),
    "sentence transformers": ("retrieval", 1.0),
    "text encoders": ("retrieval", 0.9),
    "semantic search": ("retrieval", 1.0),
    "rag": ("retrieval", 0.8),
    "information retrieval": ("retrieval", 1.0),
    "information retrieval systems": ("retrieval", 1.0),

    # Vector DBs / hybrid search infra
    "faiss": ("vectordb", 1.0),
    "pinecone": ("vectordb", 1.0),
    "weaviate": ("vectordb", 1.0),
    "qdrant": ("vectordb", 1.0),
    "milvus": ("vectordb", 1.0),
    "opensearch": ("vectordb", 1.0),
    "elasticsearch": ("vectordb", 1.0),
    "pgvector": ("vectordb", 0.9),
    "search backend": ("vectordb", 0.9),
    "search infrastructure": ("vectordb", 0.9),
    "search & discovery": ("vectordb", 0.8),
    "bm25": ("vectordb", 0.9),
    "haystack": ("vectordb", 0.8),
    "llamaindex": ("vectordb", 0.6),
    "langchain": ("vectordb", 0.4),  # JD: "framework enthusiasts" caution -> lower weight

    # Python / core ML
    "python": ("core", 1.0),
    "pytorch": ("core", 0.8),
    "tensorflow": ("core", 0.6),
    "scikit-learn": ("core", 0.5),
    "machine learning": ("core", 0.6),
    "deep learning": ("core", 0.6),
    "nlp": ("core", 0.7),
    "natural language processing": ("core", 0.7),

    # Ranking / eval
    "learning to rank": ("ranking_eval", 1.0),
    "ranking systems": ("ranking_eval", 1.0),
    "recommendation systems": ("ranking_eval", 0.9),
    "content matching": ("ranking_eval", 0.8),
    "indexing algorithms": ("ranking_eval", 0.7),
    "feature engineering": ("ranking_eval", 0.5),

    # Nice-to-have: fine-tuning
    "fine-tuning llms": ("finetune", 0.6),
    "lora": ("finetune", 0.6),
    "qlora": ("finetune", 0.6),
    "peft": ("finetune", 0.6),
    "prompt engineering": ("finetune", 0.3),
    "llms": ("finetune", 0.4),

    # Distributed / infra (nice to have)
    "kubernetes": ("infra", 0.3),
    "mlops": ("infra", 0.4),
    "mlflow": ("infra", 0.3),
    "open-source ml libraries": ("infra", 0.3),
    "weights & biases": ("infra", 0.3),
    "kubeflow": ("infra", 0.3),
}

# Skills that signal CV/speech/robotics-only specialization, which the JD
# explicitly deprioritizes unless paired with NLP/IR skills.
CV_SPEECH_SKILLS = {
    "computer vision", "opencv", "image classification", "object detection",
    "yolo", "cnn", "diffusion models", "gans", "asr", "tts",
    "speech recognition",
}


# ---------------------------------------------------------------------------
# 3. Company / industry classification — product company vs. consulting
# ---------------------------------------------------------------------------

# Substrings (lowercased) that mark a company as a pure-services / consulting
# firm per the JD's explicit list.
CONSULTING_FIRM_MARKERS = [
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "mindtree", "hcl", "tech mahindra", "ibm consulting", "deloitte",
]

# Industry strings treated as "IT services" / consulting context.
CONSULTING_INDUSTRIES = {"it services", "consulting"}


# ---------------------------------------------------------------------------
# 4. Location classification
# ---------------------------------------------------------------------------

# JD: Pune/Noida preferred; Hyderabad, Mumbai, Delhi NCR welcome.
PREFERRED_CITIES = {
    "pune", "noida",
}
WELCOME_CITIES = {
    "hyderabad", "mumbai", "delhi", "gurugram", "gurgaon", "ncr",
}
# Any other India location gets a smaller, nonzero score (relocation plausible).

LOCATION_SCORES = {
    "preferred": 1.0,
    "welcome": 0.85,
    "other_india": 0.55,
    "outside_india_relocate": 0.35,
    "outside_india_no_relocate": 0.05,
}
