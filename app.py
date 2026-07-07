"""
==========================================================
SkillSync - Intelligent Resume Skill Gap Analyzer
Word2Vec + Clean Skill Matching
==========================================================
"""

import os
import re
import pickle
import streamlit as st
from gensim.models import Word2Vec

# ----------------------------------------------------
# Page Config
# ----------------------------------------------------

st.set_page_config(
    page_title="SkillSync",
    page_icon="🧩",
    layout="wide"
)

st.title("🧩 SkillSync")
st.caption("AI Resume vs Job Description Skill Gap Analyzer")

# ----------------------------------------------------
# File Paths
# ----------------------------------------------------

MODEL_PATH = "skillsync_word2vec.model"
SKILL_PATH = "final_skill_list.pkl"

# ----------------------------------------------------
# Text Cleaning
# ----------------------------------------------------

def clean_text(text):

    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(r"http\S+", " ", text)

    text = re.sub(
        r"[^a-zA-Z0-9+#. ]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ----------------------------------------------------
# Load Word2Vec
# ----------------------------------------------------

@st.cache_resource
def load_model():

    try:

        return Word2Vec.load(MODEL_PATH)

    except Exception as e:

        st.error(f"Unable to load model\n\n{e}")

        st.stop()


# ----------------------------------------------------
# Load Skill Vocabulary
# ----------------------------------------------------

@st.cache_resource
def load_skill_vocab():

    try:

        with open(SKILL_PATH, "rb") as f:

            skills = pickle.load(f)

        skills = sorted({

            clean_text(i)

            for i in skills

            if len(clean_text(i)) >= 3

        })

        return skills

    except Exception as e:

        st.error(e)

        st.stop()


# ----------------------------------------------------
# Initialize
# ----------------------------------------------------

model = load_model()

skill_vocab = load_skill_vocab()

st.success(
    f"Loaded {len(skill_vocab)} skills."
)

# ----------------------------------------------------
# Aliases
# ----------------------------------------------------

ALIASES = {

    "ml":"machine learning",

    "dl":"deep learning",

    "ai":"artificial intelligence",

    "py":"python",

    "js":"javascript",

    "ts":"typescript",

    "postgres":"postgresql",

    "mongo":"mongodb",

    "node.js":"nodejs",

    "react.js":"react",

    "next.js":"nextjs",

    "express.js":"express",

    "powerbi":"power bi",

    "aws cloud":"aws",

    "azure cloud":"azure",

    "google cloud":"gcp"

}

# ----------------------------------------------------
# Remove Noise
# ----------------------------------------------------

STOP_WORDS = {

    "ability","abilities","working","work",

    "company","client","team","knowledge",

    "required","preferred","good","strong",

    "excellent","support","assist","maintain",

    "develop","developing","analysis",

    "computer","science","engineering",

    "engineer","engineers",

    "technology","technologies",

    "business","system","systems",

    "project","projects","application",

    "applications","field","solution",

    "solutions","using","use"

}

skill_vocab = sorted({

    s

    for s in skill_vocab

    if (

        len(s) >= 3

        and len(s.split()) <= 3

        and s not in STOP_WORDS

        and not s.isdigit()

    )

})
# ----------------------------------------------------
# Exact Skill Extraction
# ----------------------------------------------------

def extract_exact_skills(text):

    cleaned = clean_text(text)

    found = set()

    for skill in skill_vocab:

        s = clean_text(skill)

        if s in STOP_WORDS:
            continue

        if len(s) < 3:
            continue

        # Exact phrase match
        if re.search(rf"\b{re.escape(s)}\b", cleaned):
            found.add(skill)
            continue

        # Alias match
        for alias, actual in ALIASES.items():

            if actual == s:

                if re.search(rf"\b{re.escape(alias)}\b", cleaned):

                    found.add(skill)

    return found


# ----------------------------------------------------
# Semantic Matching (Word2Vec)
# ----------------------------------------------------

def semantic_expand(skills,
                    threshold=0.70):

    expanded = set(skills)

    if model is None:
        return expanded

    for skill in list(skills):

        words = clean_text(skill).split()

        for word in words:

            if word not in model.wv:
                continue

            try:

                neighbours = model.wv.most_similar(
                    word,
                    topn=10
                )

            except Exception:
                continue

            for neighbour, score in neighbours:

                if score < threshold:
                    continue

                for vocab_skill in skill_vocab:

                    vocab_words = clean_text(
                        vocab_skill
                    ).split()

                    if neighbour in vocab_words:

                        expanded.add(vocab_skill)

    return expanded


# ----------------------------------------------------
# Resume Skill Extraction
# ----------------------------------------------------

def extract_resume_skills(text):

    skills = extract_exact_skills(text)

    skills = semantic_expand(skills)

    return sorted(skills)


# ----------------------------------------------------
# JD Skill Extraction
# ----------------------------------------------------

def extract_jd_skills(text):

    skills = extract_exact_skills(text)

    return sorted(skills)
    # ----------------------------------------------------
# Skill Gap Analysis
# ----------------------------------------------------

def analyze_skill_gap(resume_text, jd_text):

    resume_skills = set(extract_resume_skills(resume_text))
    jd_skills = set(extract_jd_skills(jd_text))

    # Remove noisy entries
    resume_skills = {
        s for s in resume_skills
        if (
            len(s) >= 3
            and s not in STOP_WORDS
            and not s.isdigit()
        )
    }

    jd_skills = {
        s for s in jd_skills
        if (
            len(s) >= 3
            and s not in STOP_WORDS
            and not s.isdigit()
        )
    }

    matched = sorted(resume_skills.intersection(jd_skills))
    missing = sorted(jd_skills - resume_skills)
    extra = sorted(resume_skills - jd_skills)

    if len(jd_skills) == 0:
        score = 0
    else:
        score = round(
            (len(matched) / len(jd_skills)) * 100,
            1
        )

    return {

        "resume": sorted(resume_skills),

        "jd": sorted(jd_skills),

        "matched": matched,

        "missing": missing,

        "extra": extra,

        "score": score

    }


# ----------------------------------------------------
# Debug
# ----------------------------------------------------

def debug_output(result):

    with st.expander("🔍 Debug Output"):

        st.write("Resume Skills")

        st.write(result["resume"])

        st.write("JD Skills")

        st.write(result["jd"])


# ----------------------------------------------------
# Recommendation
# ----------------------------------------------------

def get_recommendation(score):

    if score >= 90:

        return (
            "Excellent Match ✅",
            "Your resume strongly matches the job description."
        )

    elif score >= 75:

        return (
            "Good Match 👍",
            "Your profile is suitable. Improve the missing skills."
        )

    elif score >= 50:

        return (
            "Average Match ⚠️",
            "Several important skills are missing."
        )

    else:

        return (
            "Low Match ❌",
            "You need to learn more required skills before applying."
        )
        # ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

with st.sidebar:

    st.header("About SkillSync")

    st.info(
        """
SkillSync compares your Resume
with a Job Description using

• Exact Skill Matching

• Word2Vec Semantic Matching

• Skill Gap Analysis
"""
    )

    st.divider()

    st.write(f"Loaded Skills : {len(skill_vocab)}")


# ----------------------------------------------------
# Input
# ----------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("📄 Resume")

    resume_text = st.text_area(

        "Paste Resume",

        height=350,

        placeholder="Paste Resume here..."

    )

with col2:

    st.subheader("💼 Job Description")

    jd_text = st.text_area(

        "Paste Job Description",

        height=350,

        placeholder="Paste Job Description here..."

    )

st.divider()

analyze = st.button(

    "🔍 Analyze Skill Gap",

    type="primary",

    use_container_width=True

)

# ----------------------------------------------------
# Analysis
# ----------------------------------------------------

if analyze:

    if resume_text.strip() == "":

        st.warning("Please paste Resume.")

        st.stop()

    if jd_text.strip() == "":

        st.warning("Please paste Job Description.")

        st.stop()

    with st.spinner("Analyzing Resume..."):

        result = analyze_skill_gap(

            resume_text,

            jd_text

        )

    st.divider()

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(

        "JD Skills",

        len(result["jd"])

    )

    m2.metric(

        "Matched",

        len(result["matched"])

    )

    m3.metric(

        "Missing",

        len(result["missing"])

    )

    m4.metric(

        "Match Score",

        f"{result['score']} %"

    )

    st.progress(result["score"]/100)

    title, message = get_recommendation(result["score"])

    if result["score"] >= 75:

        st.success(title)

        st.write(message)

    elif result["score"] >= 50:

        st.warning(title)

        st.write(message)

    else:

        st.error(title)

        st.write(message)

# ----------------------------------------------------
# Skills
# ----------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.subheader("✅ Matched Skills")

        if result["matched"]:

            for skill in result["matched"]:

                st.success(skill)

        else:

            st.info("No Matched Skills")

    with c2:

        st.subheader("❌ Missing Skills")

        if result["missing"]:

            for skill in result["missing"]:

                st.error(skill)

        else:

            st.success("No Missing Skills")

    with c3:

        st.subheader("⭐ Extra Skills")

        if result["extra"]:

            for skill in result["extra"]:

                st.info(skill)

        else:

            st.write("No Extra Skills")

# ----------------------------------------------------
# Debug
# ----------------------------------------------------

    debug_output(result)

# ----------------------------------------------------
# Report
# ----------------------------------------------------

    report = f"""

SkillSync Report

==========================

Resume Skills

{', '.join(result['resume'])}

--------------------------

JD Skills

{', '.join(result['jd'])}

--------------------------

Matched Skills

{', '.join(result['matched'])}

--------------------------

Missing Skills

{', '.join(result['missing'])}

--------------------------

Extra Skills

{', '.join(result['extra'])}

--------------------------

Overall Score

{result['score']} %

"""

    st.download_button(

        "⬇ Download Report",

        report,

        file_name="SkillSync_Report.txt",

        mime="text/plain"

    )

st.divider()

st.caption("SkillSync • AI Resume Skill Gap Analyzer")
