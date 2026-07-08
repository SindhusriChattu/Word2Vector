import streamlit as st
import pickle
import re
import numpy as np
import spacy
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(page_title="SkillSync - Resume Skill Gap Analyzer", layout="wide")

# ----------------------------------------------------------------------
# LOAD ARTIFACTS (cached so they only load once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    w2v_model = Word2Vec.load("skillsync_word2vec.model")

    with open("final_skill_list.pkl", "rb") as f:
        final_skill_list = pickle.load(f)

    with open("ner_candidates.pkl", "rb") as f:
        ner_candidates = pickle.load(f)

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import sys, subprocess, os
        MODEL_DIR = "/tmp/spacy_models"
        if not os.path.exists(os.path.join(MODEL_DIR, "en_core_web_sm")):
            subprocess.run(
                [
                    sys.executable, "-m", "pip", "install",
                    "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl",
                    "--target", MODEL_DIR,
                    "--no-deps",
                ],
                check=True,
            )
        if MODEL_DIR not in sys.path:
            sys.path.insert(0, MODEL_DIR)
        import en_core_web_sm
        nlp = en_core_web_sm.load()

    return w2v_model, final_skill_list, ner_candidates, nlp


w2v_model, final_skill_list, ner_candidates, nlp = load_artifacts()

# Normalize the master skill list once (lowercase, stripped)
MASTER_SKILLS = sorted(set(s.strip().lower() for s in final_skill_list if s and s.strip()))

# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+\#\.\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_candidates(text: str, nlp_model):
    """Extract noun-phrase / entity candidates from raw text using spaCy,
    mirroring the logic used to build ner_candidates.pkl during training."""
    doc = nlp_model(text)
    candidates = set()

    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip().lower()
        phrase = re.sub(r"[^a-z0-9\s\+\#\.\-]", "", phrase).strip()
        if 1 < len(phrase) <= 40:
            candidates.add(phrase)

    for ent in doc.ents:
        if ent.label_ in ("ORG", "PRODUCT", "LANGUAGE", "SKILL"):
            phrase = ent.text.strip().lower()
            if 1 < len(phrase) <= 40:
                candidates.add(phrase)

    return candidates


def match_against_master_list(candidates: set, fuzzy_threshold: float = 0.86) -> set:
    """Exact match first, then fuzzy match (handles typos, hyphen/space
    variants, plurals like 'pythons' vs 'python') using difflib."""
    import difflib
    matched = set()
    for cand in candidates:
        if cand in MASTER_SKILLS:
            matched.add(cand)
            continue

        token_matched = False
        for token in cand.split():
            if token in MASTER_SKILLS:
                matched.add(token)
                token_matched = True
        if token_matched:
            continue

        # Fuzzy fallback: catches near-misses like "power-bi" vs "power bi"
        close = difflib.get_close_matches(
            cand.replace("-", " "), MASTER_SKILLS, n=1, cutoff=fuzzy_threshold
        )
        if close:
            matched.add(close[0])

    return matched


def semantic_match(skill: str, target_skills: set, model, threshold: float = 0.55):
    """Check if `skill` is semantically close (via Word2Vec) to anything
    in target_skills. Returns the best matching skill and its similarity
    score, or (None, 0.0) if nothing clears the threshold."""
    if skill not in model.wv:
        return None, 0.0

    best_match, best_score = None, 0.0
    for target in target_skills:
        if target not in model.wv:
            continue
        score = model.wv.similarity(skill, target)
        if score > best_score:
            best_match, best_score = target, score

    if best_score >= threshold:
        return best_match, best_score
    return None, 0.0


def cluster_skills(skills: list, model, n_clusters: int = 3):
    """Group a list of skills into n_clusters using KMeans on their
    Word2Vec vectors. Skills without a vector are grouped separately."""
    vectorizable = [s for s in skills if s in model.wv]
    unvectorizable = [s for s in skills if s not in model.wv]

    if len(vectorizable) < n_clusters:
        n_clusters = max(1, len(vectorizable))

    clusters = {}
    if vectorizable:
        vectors = np.array([model.wv[s] for s in vectorizable])
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(vectors)
        for skill, label in zip(vectorizable, labels):
            clusters.setdefault(f"Group {label + 1}", []).append(skill)

    if unvectorizable:
        clusters.setdefault("Other / Uncommon Terms", []).extend(unvectorizable)

    return clusters


def calculate_ats_score(matched_exact: set, semantic_covered: dict, jd_skills: set) -> float:
    """Weighted ATS score out of 100. Exact matches count fully;
    semantically-covered skills count partially since the wording
    doesn't exactly match what an ATS keyword scanner looks for."""
    total_jd = len(jd_skills)
    if total_jd == 0:
        return 0.0

    EXACT_WEIGHT = 1.0
    SEMANTIC_WEIGHT = 0.7

    raw_score = (len(matched_exact) * EXACT_WEIGHT) + (len(semantic_covered) * SEMANTIC_WEIGHT)
    score = (raw_score / total_jd) * 100
    return round(min(score, 100), 1)


def get_shortlist_verdict(ats_score: float):
    """Returns (label, color, message) based on ATS score band.
    NOTE: this is a rule-based heuristic, not a trained ML prediction —
    there's no labeled hiring-outcome dataset behind this, so treat it
    as a directional signal, not a guarantee."""
    if ats_score >= 80:
        return "🟢 High Shortlist Chance", "green", "Strong keyword alignment with this JD."
    elif ats_score >= 60:
        return "🟡 Moderate Shortlist Chance", "orange", "Decent match, but some important skills are missing."
    elif ats_score >= 40:
        return "🟠 Low Shortlist Chance", "orange", "Noticeable skill gaps — consider tailoring your resume to this JD."
    else:
        return "🔴 High Reject Risk", "red", "Resume doesn't align well with this JD's key requirements."


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🔍 SkillSync — Resume vs Job Description Skill Gap Analyzer")
st.caption("Paste your resume and a target job description to see matched skills and gaps.")

col1, col2 = st.columns(2)
with col1:
    resume_text = st.text_area("📄 Paste Resume Text", height=300, placeholder="Paste your resume content here...")
with col2:
    jd_text = st.text_area("💼 Paste Job Description Text", height=300, placeholder="Paste the job description here...")

n_clusters = st.slider("Number of skill gap groups", min_value=2, max_value=6, value=3)

analyze_btn = st.button("Analyze Skill Gap", type="primary")

if analyze_btn:
    if not resume_text.strip() or not jd_text.strip():
        st.warning("Please paste both resume text and job description text before analyzing.")
    else:
        with st.spinner("Extracting and matching skills..."):
            resume_candidates = extract_candidates(clean_text(resume_text), nlp)
            jd_candidates = extract_candidates(clean_text(jd_text), nlp)

            resume_skills = match_against_master_list(resume_candidates)
            jd_skills = match_against_master_list(jd_candidates)

            # Exact overlap
            matched_exact = resume_skills & jd_skills

            # Skills required by JD but missing from resume (raw gap)
            raw_gap = jd_skills - resume_skills

            # For each gap skill, check if a semantically similar skill
            # already exists in the resume (covers synonyms / related tools)
            semantic_covered = {}
            true_gap = []
            for skill in raw_gap:
                match, score = semantic_match(skill, resume_skills, w2v_model)
                if match:
                    semantic_covered[skill] = (match, score)
                else:
                    true_gap.append(skill)

        st.divider()

        m1, m2, m3 = st.columns(3)
        m1.metric("Skills in JD", len(jd_skills))
        m2.metric("Matched Skills", len(matched_exact) + len(semantic_covered))
        m3.metric("Real Skill Gap", len(true_gap))

        # ---- ATS SCORE + SHORTLIST VERDICT ----
        ats_score = calculate_ats_score(matched_exact, semantic_covered, jd_skills)
        verdict_label, verdict_color, verdict_msg = get_shortlist_verdict(ats_score)

        st.divider()
        st.subheader("🎯 ATS Match Score")
        sc1, sc2 = st.columns([1, 2])
        with sc1:
            st.metric("ATS Score", f"{ats_score}%")
        with sc2:
            st.progress(int(ats_score))
        st.markdown(f"### {verdict_label}")
        st.caption(verdict_msg)
        st.caption(
            "⚠️ This is a rule-based keyword-match estimate, not a guaranteed hiring outcome — "
            "actual ATS systems and recruiters weigh many other factors too."
        )

        st.subheader("✅ Matched Skills (exact)")
        st.write(", ".join(sorted(matched_exact)) if matched_exact else "None found.")

        if semantic_covered:
            st.subheader("🔁 Semantically Covered (similar skill found in resume)")
            for skill, (match, score) in sorted(semantic_covered.items()):
                st.write(f"- **{skill}** ≈ your **{match}** (similarity: {score:.2f})")

        st.subheader("❌ Skill Gap (missing from resume)")
        if true_gap:
            st.write(", ".join(sorted(true_gap)))
        else:
            st.success("No major skill gaps found — great match!")

        if true_gap:
            st.subheader("📊 Suggested Skill Gap Groups")
            groups = cluster_skills(true_gap, w2v_model, n_clusters=n_clusters)
            for group_name, group_skills in groups.items():
                st.markdown(f"**{group_name}:** {', '.join(sorted(group_skills))}")

        st.divider()
        with st.expander("Debug: raw extracted candidates"):
            st.write("Resume candidates:", sorted(resume_candidates))
            st.write("JD candidates:", sorted(jd_candidates))
