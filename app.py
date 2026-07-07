"""
==========================================================
SkillSync - Intelligent Resume vs Job Description Analyzer
Word2Vec + Canonical Skill Vocabulary
==========================================================
"""

import os
import re
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st

from gensim.models import Word2Vec


# --------------------------------------------------------
# Streamlit Page
# --------------------------------------------------------

st.set_page_config(
    page_title="SkillSync",
    page_icon="🧩",
    layout="wide"
)

st.title("🧩 SkillSync")
st.caption(
    "AI Powered Resume Skill Gap Analyzer using Word2Vec + NLP"
)

# --------------------------------------------------------
# Paths
# --------------------------------------------------------

MODEL_PATH = "skillsync_word2vec.model"
SKILL_PATH = "final_skill_list.pkl"
NER_PATH = "ner_candidates.pkl"

# --------------------------------------------------------
# Text Cleaning
# --------------------------------------------------------

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.lower()

    text = re.sub(r"http\\S+", " ", text)

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


def tokenize(text):

    return clean_text(text).split()

# --------------------------------------------------------
# Load Word2Vec
# --------------------------------------------------------

@st.cache_resource
def load_model():

    try:

        model = Word2Vec.load(MODEL_PATH)

        return model

    except Exception as e:

        st.error(
            f"""
Unable to load Word2Vec model.

{e}
"""
        )

        st.stop()

# --------------------------------------------------------
# Load Canonical Skills
# --------------------------------------------------------

@st.cache_resource
def load_skill_vocab():

    try:

        with open(SKILL_PATH, "rb") as f:

            skills = pickle.load(f)

        skills = sorted(

            {

                clean_text(s)

                for s in skills

                if len(clean_text(s)) > 1

            }

        )

        return skills

    except Exception as e:

        st.error(e)

        st.stop()

# --------------------------------------------------------
# Optional NER candidates
# --------------------------------------------------------

@st.cache_resource
def load_candidates():

    if not os.path.exists(NER_PATH):

        return []

    try:

        with open(NER_PATH, "rb") as f:

            data = pickle.load(f)

        return list(

            {

                clean_text(i)

                for i in data

                if len(clean_text(i)) > 1

            }

        )

    except:

        return []

# --------------------------------------------------------
# Initialize
# --------------------------------------------------------

model = load_model()

skill_vocab = load_skill_vocab()

ner_candidates = load_candidates()

skill_set = set(skill_vocab)

st.success(
    f"Loaded {len(skill_vocab)} canonical skills."
)

# --------------------------------------------------------
# Common Skill Aliases
# --------------------------------------------------------

ALIASES = {

    "ml":"machine learning",

    "dl":"deep learning",

    "ai":"artificial intelligence",

    "js":"javascript",

    "ts":"typescript",

    "py":"python",

    "postgres":"postgresql",

    "mongo":"mongodb",

    "powerbi":"power bi",

    "node":"nodejs",

    "node.js":"nodejs",

    "react.js":"react",

    "next.js":"nextjs",

    "express.js":"express",

    "c sharp":"c#",

    "dot net":".net",

    "azure cloud":"azure",

    "aws cloud":"aws",

    "google cloud":"gcp",

    "google cloud platform":"gcp"

}
# --------------------------------------------------------
# Exact Skill Extraction
# --------------------------------------------------------
def extract_exact_skills(text):

    cleaned = clean_text(text)

    tokens = set(cleaned.split())

    found = set()

    for skill in skill_vocab:

        s = clean_text(skill)

        if not s:
            continue

        if len(s) < 3:
            continue

        if s in STOP_WORDS:
            continue

        if s in ALIASES:
            s = ALIASES[s]

        if s in cleaned:
            found.add(skill)
            continue

        words = s.split()

        if all(w in tokens for w in words):
            found.add(skill)

    return found



# --------------------------------------------------------
# Alias Expansion
# --------------------------------------------------------

def expand_aliases(skills):

    expanded = set(skills)

    for skill in list(skills):

        s = clean_text(skill)

        if s in ALIASES:

            expanded.add(ALIASES[s])

    return expanded


# --------------------------------------------------------
# Word2Vec Semantic Expansion
# --------------------------------------------------------

def semantic_expand(skills,
                    similarity=0.55):

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
                    topn=15
                )

            except Exception:
                continue

            for neighbour, score in neighbours:

                if score < similarity:
                    continue

                for vocab_skill in skill_vocab:

                    vocab_words = clean_text(
                        vocab_skill
                    ).split()

                    if neighbour in vocab_words:

                        expanded.add(vocab_skill)

    return expanded


# --------------------------------------------------------
# Resume Skill Extraction
# --------------------------------------------------------

def extract_resume_skills(text):

    skills = extract_exact_skills(text)

    skills |= match_ner_candidates(text)

    skills = expand_aliases(skills)

    skills = semantic_expand(skills)

    return sorted(skills)


# --------------------------------------------------------
# JD Skill Extraction
# --------------------------------------------------------

def extract_jd_skills(text):

    skills = extract_exact_skills(text)

    skills |= match_ner_candidates(text)

    skills = expand_aliases(skills)

    return sorted(skills)


# --------------------------------------------------------
# Skill Gap Analysis
# --------------------------------------------------------

def analyze_skill_gap(resume_text,
                      jd_text):

    resume_skills = set(
        extract_resume_skills(
            resume_text
        )
    )

    jd_skills = set(
        extract_jd_skills(
            jd_text
        )
    )

    matched = sorted(
        resume_skills &
        jd_skills
    )

    missing = sorted(
        jd_skills -
        resume_skills
    )

    extra = sorted(
        resume_skills -
        jd_skills
    )

    if len(jd_skills):

        score = round(

            len(matched)
            /
            len(jd_skills)
            *
            100,
            1
        )

    else:

        score = 0.0

    return {

        "resume": sorted(
            resume_skills
        ),

        "jd": sorted(
            jd_skills
        ),

        "matched": matched,

        "missing": missing,

        "extra": extra,

        "score": score

    }


# --------------------------------------------------------
# Debug Helper
# --------------------------------------------------------

def debug_output(result):

    with st.expander(
        "Debug Output"
    ):

        st.write(
            "Resume Skills",
            result["resume"]
        )

        st.write(
            "JD Skills",
            result["jd"]
        )

        st.write(
            "Matched",
            result["matched"]
        )

        st.write(
            "Missing",
            result["missing"]
        )

        st.write(
            "Extra",
            result["extra"]
        )
# --------------------------------------------------------
# Sidebar
# --------------------------------------------------------

with st.sidebar:

    st.header("About SkillSync")

    st.info(
        """
Compare your Resume with a Job Description
using NLP + Word2Vec semantic similarity.

Features

✔ Exact Skill Matching

✔ Semantic Skill Matching

✔ Missing Skills

✔ Extra Skills

✔ Match Percentage
"""
    )

    st.divider()

    st.write(
        f"Canonical Skills : {len(skill_vocab)}"
    )

    st.write(
        f"NER Candidates : {len(ner_candidates)}"
    )


# --------------------------------------------------------
# Input Area
# --------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("📄 Resume")

    resume_text = st.text_area(

        "Paste Resume",

        height=350,

        placeholder="Paste your resume here..."

    )


with col2:

    st.subheader("💼 Job Description")

    jd_text = st.text_area(

        "Paste Job Description",

        height=350,

        placeholder="Paste job description here..."

    )


st.divider()


analyze = st.button(

    "🔍 Analyze Skill Gap",

    use_container_width=True,

    type="primary"

)

# --------------------------------------------------------
# Run Analysis
# --------------------------------------------------------

if analyze:

    if resume_text.strip() == "":

        st.warning(
            "Please enter Resume."
        )

        st.stop()

    if jd_text.strip() == "":

        st.warning(
            "Please enter Job Description."
        )

        st.stop()

    with st.spinner("Analyzing..."):

        result = analyze_skill_gap(

            resume_text,

            jd_text

        )

# --------------------------------------------------------
# Metrics
# --------------------------------------------------------

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

    st.progress(

        result["score"] / 100

    )

# --------------------------------------------------------
# Skill Lists
# --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.subheader("✅ Matched Skills")

        if result["matched"]:

            for skill in result["matched"]:

                st.success(skill)

        else:

            st.info("No matched skills")



    with c2:

        st.subheader("❌ Missing Skills")

        if result["missing"]:

            for skill in result["missing"]:

                st.error(skill)

        else:

            st.success("No missing skills")


    with c3:

        st.subheader("⭐ Extra Skills")

        if result["extra"]:

            for skill in result["extra"]:

                st.info(skill)

        else:

            st.write("No extra skills")

# --------------------------------------------------------
# Debug
# --------------------------------------------------------

    debug_output(result)
# --------------------------------------------------------
# Resume Summary
# --------------------------------------------------------

    st.divider()

    st.subheader("📊 Resume Summary")

    score = result["score"]

    if score >= 90:

        st.success(
            """
Excellent match!

Your resume already covers almost all required skills.
You are highly suitable for this role.
"""
        )

    elif score >= 75:

        st.success(
            """
Good Match.

Your resume satisfies most of the required skills.
Learning the missing skills can significantly improve your chances.
"""
        )

    elif score >= 50:

        st.warning(
            """
Average Match.

Several important skills are missing.
Consider improving your profile before applying.
"""
        )

    else:

        st.error(
            """
Low Match.

Your resume currently misses many important skills.

Upskilling is recommended before applying.
"""
        )

# --------------------------------------------------------
# Recommendations
# --------------------------------------------------------

    st.divider()

    st.subheader("🎯 Learning Recommendation")

    if len(result["missing"]) == 0:

        st.success(
            "No additional skills required."
        )

    else:

        st.write(
            "Recommended learning order:"
        )

        for i, skill in enumerate(result["missing"], start=1):

            st.write(
                f"{i}. {skill}"
            )

# --------------------------------------------------------
# Resume Statistics
# --------------------------------------------------------

    st.divider()

    st.subheader("📈 Statistics")

    stat1, stat2, stat3 = st.columns(3)

    stat1.metric(

        "Resume Skills",

        len(result["resume"])

    )

    stat2.metric(

        "Job Skills",

        len(result["jd"])

    )

    stat3.metric(

        "Extra Skills",

        len(result["extra"])

    )

# --------------------------------------------------------
# Download Report
# --------------------------------------------------------

    report = f"""
SkillSync Report

==================================

Resume Skills

{", ".join(result["resume"])}

----------------------------------

Job Description Skills

{", ".join(result["jd"])}

----------------------------------

Matched Skills

{", ".join(result["matched"])}

----------------------------------

Missing Skills

{", ".join(result["missing"])}

----------------------------------

Extra Skills

{", ".join(result["extra"])}

----------------------------------

Overall Match

{result["score"]} %

"""

    st.download_button(

        "⬇ Download Report",

        report,

        file_name="SkillSync_Report.txt",

        mime="text/plain"

    )

# --------------------------------------------------------
# Footer
# --------------------------------------------------------

st.divider()

st.caption(

    "SkillSync • NLP Resume Analyzer • Word2Vec + Semantic Matching"

)
