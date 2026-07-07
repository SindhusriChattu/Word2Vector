"""
SkillSync — NLP-based Job Skill Gap Analyzer
Uses a Word2Vec model trained on LinkedIn job postings + an auto-discovered
canonical skill vocabulary (built via spaCy NER + KMeans clustering, see
skillsync_word2vec_ner_clustering.ipynb) to compare a resume against a job
description and surface matched / missing skills.

Run with:
    streamlit run app.py

Expected files in the same folder as this script:
    - skillsync_word2vec.model   (gensim Word2Vec, saved with model.save(...))
    - final_skill_list.pkl       (list[str] of canonical skill names)
    - ner_candidates.pkl         (list[str] of raw NER candidates, optional -
                                   used only to enrich matching, app still
                                   works fine without it)
"""

import re
import pickle
import streamlit as st

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="SkillSync — Job Skill Gap Analyzer", page_icon="🧩", layout="wide")

MODEL_PATH = "skillsync_word2vec.model"
SKILLS_PATH = "final_skill_list.pkl"
NER_PATH = "ner_candidates.pkl"


# ------------------------------------------------------------------
# Text cleaning — mirrors the notebook's clean_text() exactly so the
# Word2Vec vocabulary lines up with whatever we tokenize here.
# ------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9+#. ]", " ", text)  # keep + # . for C++, C#, .NET
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    return clean_text(text).split()


# ------------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Word2Vec model...")
def load_word2vec(path):
    try:
        from gensim.models import Word2Vec
        return Word2Vec.load(path)
    except Exception as e:
        st.error(
            f"Couldn't load Word2Vec model from '{path}'. "
            f"Make sure the file is in the same folder as app.py.\n\nDetails: {e}"
        )
        return None


@st.cache_resource(show_spinner="Loading canonical skill vocabulary...")
def load_skill_list(path):
    try:
        with open(path, "rb") as f:
            skills = pickle.load(f)
        # de-dupe, drop empties/junk, keep order-independent set for lookups
        skills = sorted({s.strip().lower() for s in skills if s and len(s.strip()) > 1})
        return skills
    except Exception as e:
        st.error(
            f"Couldn't load skill list from '{path}'. "
            f"Make sure the file is in the same folder as app.py.\n\nDetails: {e}"
        )
        return []


@st.cache_resource(show_spinner=False)
def load_ner_candidates(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return []  # optional file — app works fine without it


# ------------------------------------------------------------------
# Core matching logic
# ------------------------------------------------------------------
def find_skills_in_text(text: str, skill_vocab):
    """
    Direct substring/phrase matching of the canonical skill vocabulary
    against the (cleaned) input text. Handles multi-word skills like
    'machine learning' as well as single tokens.
    """
    cleaned = " " + clean_text(text) + " "
    found = set()
    for skill in skill_vocab:
        skill_clean = clean_text(skill)
        if not skill_clean:
            continue
        if f" {skill_clean} " in cleaned:
            found.add(skill)
    return found


def expand_with_word2vec(found_skills, skill_vocab, w2v_model, topn=5, threshold=0.65):
    """
    For every skill already found, look up nearby vocabulary words in the
    Word2Vec embedding space and check whether any *other* canonical skill
    is close enough to count as an implied/synonymous match.
    e.g. resume says 'ml' -> model space near 'machine learning'.
    """
    if w2v_model is None:
        return set()

    implied = set()
    skill_vocab_set = set(skill_vocab)

    for skill in found_skills:
        skill_clean = clean_text(skill)
        words = skill_clean.split()
        # try to pull neighbors for each word making up the skill phrase
        for w in words:
            if w in w2v_model.wv:
                try:
                    neighbors = w2v_model.wv.most_similar(w, topn=topn)
                except Exception:
                    continue
                for neighbor_word, score in neighbors:
                    if score < threshold:
                        continue
                    for candidate_skill in skill_vocab_set:
                        if neighbor_word in clean_text(candidate_skill).split():
                            implied.add(candidate_skill)
    return implied - found_skills


def compute_gap(resume_text, jd_text, skill_vocab, w2v_model):
    resume_found = find_skills_in_text(resume_text, skill_vocab)
    jd_found = find_skills_in_text(jd_text, skill_vocab)

    # use word2vec to soften exact-match strictness (catch synonyms/variants)
    resume_implied = expand_with_word2vec(resume_found, skill_vocab, w2v_model)
    resume_effective = resume_found | resume_implied

    matched = sorted(jd_found & resume_effective)
    missing = sorted(jd_found - resume_effective)
    extra = sorted(resume_effective - jd_found)

    match_pct = round(100 * len(matched) / len(jd_found), 1) if jd_found else 0.0

    return {
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "jd_found": sorted(jd_found),
        "match_pct": match_pct,
    }


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🧩 SkillSync")
st.caption("NLP-based job skill gap analyzer — Word2Vec + NER-derived skill vocabulary")

with st.sidebar:
    st.header("About")
    st.write(
        "SkillSync compares your resume against a job description using a "
        "skill vocabulary that was auto-discovered from the LinkedIn Job "
        "Postings 2023–2024 dataset, instead of a hardcoded list."
    )
    st.markdown(
        "**Pipeline:**\n"
        "1. spaCy NER pulls candidate skill/tool entities from job posts\n"
        "2. Word2Vec is trained on the same job-posting corpus\n"
        "3. KMeans clusters similar/duplicate mentions (e.g. 'ml' & 'machine learning')\n"
        "4. Shortest term per cluster becomes the canonical skill name"
    )
    st.divider()
    st.caption(
        "⚠️ Because the NER labels include ORG/PRODUCT/GPE/LANGUAGE, the "
        "vocabulary can include some noise (company names, locations). "
        "Matches should be read as a helpful signal, not a certified score."
    )

w2v_model = load_word2vec(MODEL_PATH)
skill_vocab = load_skill_list(SKILLS_PATH)
_ = load_ner_candidates(NER_PATH)  # loaded for future use / debugging

if not skill_vocab:
    st.stop()

st.write(f"Loaded skill vocabulary: **{len(skill_vocab)}** canonical skills.")

col1, col2 = st.columns(2)
with col1:
    resume_text = st.text_area("📄 Paste your resume text", height=300, placeholder="Paste resume content here...")
with col2:
    jd_text = st.text_area("📋 Paste the job description", height=300, placeholder="Paste job description here...")

analyze = st.button("🔍 Analyze Skill Gap", type="primary", use_container_width=True)

if analyze:
    if not resume_text.strip() or not jd_text.strip():
        st.warning("Please paste both your resume text and a job description before analyzing.")
    else:
        with st.spinner("Analyzing..."):
            result = compute_gap(resume_text, jd_text, skill_vocab, w2v_model)

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Skills in JD", len(result["jd_found"]))
        m2.metric("Matched", len(result["matched"]))
        m3.metric("Missing", len(result["missing"]))
        m4.metric("Match %", f"{result['match_pct']}%")

        st.progress(min(result["match_pct"] / 100, 1.0))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("✅ Matched Skills")
            if result["matched"]:
                for s in result["matched"]:
                    st.write(f"- {s}")
            else:
                st.write("_No overlapping skills found._")

        with c2:
            st.subheader("❌ Missing Skills")
            if result["missing"]:
                for s in result["missing"]:
                    st.write(f"- {s}")
            else:
                st.write("_No gaps found — great match!_")

        with c3:
            st.subheader("➕ Extra Skills (resume only)")
            if result["extra"]:
                for s in result["extra"][:30]:
                    st.write(f"- {s}")
                if len(result["extra"]) > 30:
                    st.caption(f"...and {len(result['extra']) - 30} more")
            else:
                st.write("_Nothing extra detected._")
