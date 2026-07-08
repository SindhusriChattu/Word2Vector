import streamlit as st
import pickle
import re
import difflib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import spacy
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SkillSync AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown("""
<style>

.main{
    background:#0E1117;
}

.block-container{
    padding-top:1rem;
    padding-bottom:2rem;
}

h1,h2,h3,h4{
    color:white;
}

.metric-card{
    background:#1C1F26;
    padding:20px;
    border-radius:12px;
    border:1px solid #31333F;
    text-align:center;
}

.skill-badge{
    display:inline-block;
    background:#2563EB;
    color:white;
    padding:6px 14px;
    margin:4px;
    border-radius:18px;
    font-size:14px;
    font-weight:600;
}

.missing-badge{
    display:inline-block;
    background:#DC2626;
    color:white;
    padding:6px 14px;
    margin:4px;
    border-radius:18px;
    font-size:14px;
    font-weight:600;
}

.semantic-badge{
    display:inline-block;
    background:#F59E0B;
    color:black;
    padding:6px 14px;
    margin:4px;
    border-radius:18px;
    font-size:14px;
    font-weight:600;
}

.summary-box{
    background:#181A20;
    padding:18px;
    border-left:5px solid #2563EB;
    border-radius:10px;
}

.small-text{
    color:#9CA3AF;
    font-size:13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_artifacts():

    w2v_model = Word2Vec.load("skillsync_word2vec.model")

    with open("final_skill_list.pkl","rb") as f:
        final_skill_list = pickle.load(f)

    with open("ner_candidates.pkl","rb") as f:
        ner_candidates = pickle.load(f)

    try:
        nlp = spacy.load("en_core_web_sm")

    except:

        import subprocess
        import sys
        import os

        MODEL_DIR="/tmp/spacy"

        if not os.path.exists(MODEL_DIR):

            subprocess.run([
                sys.executable,
                "-m",
                "pip",
                "install",
                "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl",
                "--target",
                MODEL_DIR,
                "--no-deps"
            ])

        if MODEL_DIR not in sys.path:
            sys.path.append(MODEL_DIR)

        import en_core_web_sm

        nlp=en_core_web_sm.load()

    return (
        w2v_model,
        final_skill_list,
        ner_candidates,
        nlp
    )


w2v_model, final_skill_list, ner_candidates, nlp = load_artifacts()

MASTER_SKILLS = sorted(
    set(
        s.strip().lower()
        for s in final_skill_list
        if s and s.strip()
    )
)

# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text=text.lower()

    text=re.sub(
        r"[^a-z0-9\s\+\#\.\-]",
        " ",
        text
    )

    text=re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

# ============================================================
# EXTRACT CANDIDATES
# ============================================================

def extract_candidates(text):

    doc=nlp(text)

    candidates=set()

    for chunk in doc.noun_chunks:

        phrase=chunk.text.strip().lower()

        phrase=re.sub(
            r"[^a-z0-9\s\+\#\.\-]",
            "",
            phrase
        )

        if 2 <= len(phrase) <= 40:
            candidates.add(phrase)

    for ent in doc.ents:

        if ent.label_ in (
            "ORG",
            "PRODUCT",
            "LANGUAGE",
            "SKILL"
        ):

            phrase=ent.text.lower().strip()

            if 2 <= len(phrase)<=40:
                candidates.add(phrase)

    return candidates

# ============================================================
# MATCH MASTER SKILLS
# ============================================================

def match_master_skills(candidates, threshold=0.86):

    matched=set()

    for cand in candidates:

        if cand in MASTER_SKILLS:
            matched.add(cand)
            continue

        for token in cand.split():

            if token in MASTER_SKILLS:
                matched.add(token)

        close=difflib.get_close_matches(
            cand.replace("-"," "),
            MASTER_SKILLS,
            n=1,
            cutoff=threshold
        )

        if close:
            matched.add(close[0])

    return matched

# ============================================================
# SEMANTIC MATCH
# ============================================================

def semantic_match(skill,resume_skills,threshold=0.55):

    if skill not in w2v_model.wv:
        return None,0

    best_skill=None
    best_score=0

    for rs in resume_skills:

        if rs not in w2v_model.wv:
            continue

        score=w2v_model.wv.similarity(skill,rs)

        if score>best_score:

            best_skill=rs
            best_score=score

    if best_score>=threshold:
        return best_skill,best_score

    return None,0

# ============================================================
# KMEANS CLUSTERING
# ============================================================

def cluster_missing_skills(
        skills,
        n_clusters=3
):

    vectors=[]
    labels=[]

    for s in skills:

        if s in w2v_model.wv:
            vectors.append(w2v_model.wv[s])
            labels.append(s)

    if len(vectors)==0:
        return {}

    n_clusters=min(
        n_clusters,
        len(vectors)
    )

    km=KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    pred=km.fit_predict(vectors)

    groups={}

    for label,cluster in zip(labels,pred):

        groups.setdefault(
            f"Group {cluster+1}",
            []
        ).append(label)

    return groups

# ============================================================
# BADGES
# ============================================================

def show_skill_badges(skills,color="blue"):

    if color=="blue":
        css="skill-badge"

    elif color=="red":
        css="missing-badge"

    else:
        css="semantic-badge"

    badges=""

    for skill in sorted(skills):

        badges+=f"""
        <span class="{css}">
        {skill.upper()}
        </span>
        """

    st.markdown(
        badges,
        unsafe_allow_html=True
    )

# ============================================================
# ATS COLOR
# ============================================================

def ats_color(score):

    if score>=80:
        return "green"

    elif score>=60:
        return "orange"

    return "red"

# ============================================================
# CONFIDENCE LABEL
# ============================================================

def confidence_label(score):

    if score>=0.90:
        return "Very High"

    elif score>=0.80:
        return "High"

    elif score>=0.70:
        return "Medium"

    return "Low"

# ============================================================
# SECTION DIVIDER
# ============================================================

def section(title):

    st.markdown("---")
    st.subheader(title)
# ============================================================
# ATS SCORE CALCULATION
# ============================================================

def calculate_ats_score(
    matched_exact,
    semantic_matches,
    jd_skills
):
    """
    Weighted ATS score.

    Exact Match = 100%
    Semantic Match = 70%

    Final score is normalized to 100.
    """

    total = len(jd_skills)

    if total == 0:
        return 0

    exact_weight = 1.0
    semantic_weight = 0.70

    score = (
        len(matched_exact) * exact_weight +
        len(semantic_matches) * semantic_weight
    ) / total

    return round(min(score * 100, 100), 1)


# ============================================================
# ATS VERDICT
# ============================================================

def ats_verdict(score):

    if score >= 85:
        return (
            "Excellent",
            "🟢 Excellent ATS Compatibility",
            "This resume is strongly aligned with the target job description."
        )

    elif score >= 70:
        return (
            "Good",
            "🟢 Good ATS Compatibility",
            "Most technical requirements are covered."
        )

    elif score >= 55:
        return (
            "Moderate",
            "🟡 Moderate ATS Compatibility",
            "The resume is suitable but still misses some important skills."
        )

    elif score >= 40:
        return (
            "Low",
            "🟠 Low ATS Compatibility",
            "Several important skills are missing."
        )

    else:
        return (
            "Poor",
            "🔴 Poor ATS Compatibility",
            "Major improvements are required before applying."
        )


# ============================================================
# SKILL CATEGORIES
# ============================================================

SKILL_CATEGORIES = {

    "Programming":[
        "python","java","c","c++","r","scala","javascript"
    ],

    "Database":[
        "sql","mysql","postgresql","oracle","mongodb","sqlite"
    ],

    "Machine Learning":[
        "machine learning",
        "deep learning",
        "tensorflow",
        "keras",
        "scikit learn",
        "xgboost",
        "lightgbm",
        "pytorch"
    ],

    "Data Analysis":[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "statistics",
        "excel"
    ],

    "Visualization":[
        "power bi",
        "tableau",
        "looker",
        "plotly"
    ],

    "Cloud":[
        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes"
    ],

    "Data Engineering":[
        "spark",
        "hadoop",
        "etl",
        "airflow",
        "kafka",
        "databricks"
    ]
}


# ============================================================
# CATEGORY BREAKDOWN
# ============================================================

def category_breakdown(resume_skills):

    report = {}

    for category, skill_list in SKILL_CATEGORIES.items():

        matched = []

        for skill in skill_list:

            if skill in resume_skills:
                matched.append(skill)

        report[category] = matched

    return report


# ============================================================
# MISSING SKILL PRIORITY
# ============================================================

HIGH_PRIORITY = {

    "python",
    "sql",
    "machine learning",
    "deep learning",
    "aws",
    "etl",
    "spark",
    "pandas",
    "power bi",
    "tableau",
    "git"
}


def skill_priority(skill):

    if skill in HIGH_PRIORITY:
        return "High"

    if len(skill.split()) >= 2:
        return "Medium"

    return "Low"


# ============================================================
# LEARNING RECOMMENDATION
# ============================================================

def recommendation(skill):

    recommendations = {

        "machine learning":
            "Add at least one ML project to your resume.",

        "deep learning":
            "Mention ANN, CNN or NLP projects.",

        "aws":
            "Include AWS Cloud or deployment experience.",

        "etl":
            "Highlight ETL pipelines or data integration work.",

        "spark":
            "Mention Apache Spark or PySpark projects.",

        "docker":
            "Add containerization experience.",

        "git":
            "Mention version control usage.",

        "power bi":
            "Show dashboard development experience.",

        "tableau":
            "Include visualization projects.",

        "sql":
            "Highlight SQL optimization and joins.",

        "python":
            "Mention automation or ML projects."
    }

    return recommendations.get(
        skill,
        "Consider adding this skill if you have relevant experience."
    )


# ============================================================
# RECRUITER SUMMARY
# ============================================================

def recruiter_summary(
    ats_score,
    matched_exact,
    semantic_matches,
    missing
):

    summary = []

    if ats_score >= 80:

        summary.append(
            "The resume demonstrates strong alignment with the target job description."
        )

    elif ats_score >= 60:

        summary.append(
            "The resume has a good technical foundation but can be strengthened."
        )

    else:

        summary.append(
            "The resume requires significant improvements before applying."
        )

    if matched_exact:

        summary.append(
            f"{len(matched_exact)} required technical skills were matched exactly."
        )

    if semantic_matches:

        summary.append(
            f"{len(semantic_matches)} additional skills were identified through semantic matching."
        )

    if missing:

        summary.append(
            f"{len(missing)} important skills are currently absent from the resume."
        )

    summary.append(
        "Adding relevant projects, certifications, and quantified achievements will improve ATS performance."
    )

    return " ".join(summary)


# ============================================================
# IMPROVEMENT TIPS
# ============================================================

def improvement_tips(missing_skills):

    tips = []

    for skill in missing_skills:

        tips.append(recommendation(skill))

    tips = list(dict.fromkeys(tips))

    return tips


# ============================================================
# RECRUITER READINESS
# ============================================================

def recruiter_readiness(score):

    if score >= 85:
        return "Excellent"

    elif score >= 70:
        return "Good"

    elif score >= 55:
        return "Moderate"

    elif score >= 40:
        return "Needs Improvement"

    return "Not Ready"


# ============================================================
# METRIC DATAFRAME
# ============================================================

def metrics_dataframe(
    matched,
    semantic,
    missing
):

    return pd.DataFrame({

        "Category":[
            "Matched Skills",
            "Semantic Matches",
            "Missing Skills"
        ],

        "Count":[
            len(matched),
            len(semantic),
            len(missing)
        ]

    })


# ============================================================
# ATS BAR CHART
# ============================================================

def ats_chart(df):

    fig = px.bar(

        df,

        x="Category",

        y="Count",

        text="Count",

        color="Category",

        height=350

    )

    fig.update_layout(

        showlegend=False,

        xaxis_title="",

        yaxis_title="",

        plot_bgcolor="#0E1117",

        paper_bgcolor="#0E1117",

        font=dict(color="white")

    )

    return fig


# ============================================================
# DONUT CHART
# ============================================================

def skill_donut(
    matched,
    semantic,
    missing
):

    fig = go.Figure(

        data=[

            go.Pie(

                labels=[
                    "Matched",
                    "Semantic",
                    "Missing"
                ],

                values=[
                    len(matched),
                    len(semantic),
                    len(missing)
                ],

                hole=0.60

            )

        ]

    )

    fig.update_layout(

        paper_bgcolor="#0E1117",

        font=dict(color="white"),

        height=400

    )

    return fig
    # ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.image(
        "https://img.icons8.com/fluency/96/resume.png",
        width=70
    )

    st.title("SkillSync AI")

    st.markdown(
        """
Analyze your resume against a Job Description using

- ✅ NLP
- ✅ Word2Vec
- ✅ Semantic Matching
- ✅ ATS Scoring
- ✅ Skill Gap Analysis
        """
    )

    st.divider()

    st.subheader("⚙ Analysis Settings")

    similarity_threshold = st.slider(
        "Semantic Match Threshold",
        min_value=0.40,
        max_value=0.90,
        value=0.55,
        step=0.05,
        help="Higher values require stronger semantic similarity."
    )

    cluster_count = st.slider(
        "Skill Gap Groups",
        min_value=2,
        max_value=6,
        value=3
    )

    st.divider()

    st.caption("SkillSync AI v2.0")
    st.caption("Powered by NLP + Word2Vec")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
# 🔍 SkillSync AI

### Professional Resume Compatibility Analyzer

Analyze your resume against a target Job Description using
**Natural Language Processing**, **Word2Vec**, and **Semantic Skill Matching**.

---
"""
)


# ============================================================
# INPUT SECTION
# ============================================================

left, right = st.columns(2)


with left:

    st.markdown("## 📄 Resume")

    resume_text = st.text_area(

        "",

        height=420,

        placeholder="""
Paste your Resume here...

Example

Education

Projects

Skills

Experience

Certifications

Achievements
"""
    )


with right:

    st.markdown("## 💼 Job Description")

    jd_text = st.text_area(

        "",

        height=420,

        placeholder="""
Paste the Job Description here...

Responsibilities

Required Skills

Preferred Skills

Qualifications
"""
    )


# ============================================================
# ACTION BUTTON
# ============================================================

st.write("")

col1, col2, col3 = st.columns([1,2,1])

with col2:

    analyze = st.button(

        "🚀 Analyze Resume",

        use_container_width=True,

        type="primary"

    )


# ============================================================
# VALIDATION
# ============================================================

if analyze:

    if not resume_text.strip():

        st.error("Please paste your Resume.")

        st.stop()

    if not jd_text.strip():

        st.error("Please paste the Job Description.")

        st.stop()


    with st.spinner("Analyzing Resume..."):

        resume_candidates = extract_candidates(
            clean_text(resume_text)
        )

        jd_candidates = extract_candidates(
            clean_text(jd_text)
        )

        resume_skills = match_master_skills(
            resume_candidates
        )

        jd_skills = match_master_skills(
            jd_candidates
        )

        matched_exact = resume_skills & jd_skills

        raw_gap = jd_skills - resume_skills

        semantic_matches = {}

        true_gap = []

        for skill in raw_gap:

            match, score = semantic_match(

                skill,

                resume_skills,

                threshold=similarity_threshold

            )

            if match:

                semantic_matches[skill] = (

                    match,

                    score

                )

            else:

                true_gap.append(skill)
    # ============================================================
    # ATS SCORE
    # ============================================================

    ats_score = calculate_ats_score(
        matched_exact,
        semantic_matches,
        jd_skills
    )

    strength, verdict, verdict_msg = ats_verdict(
        ats_score
    )

    readiness = recruiter_readiness(
        ats_score
    )

    section("📊 Resume Compatibility Report")

    # ============================================================
    # KPI CARDS
    # ============================================================

    k1, k2, k3, k4 = st.columns(4)

    with k1:

        st.metric(
            "ATS Score",
            f"{ats_score}%"
        )

    with k2:

        st.metric(
            "Matched Skills",
            len(matched_exact)
        )

    with k3:

        st.metric(
            "Semantic Matches",
            len(semantic_matches)
        )

    with k4:

        st.metric(
            "Missing Skills",
            len(true_gap)
        )

    st.write("")

    # ============================================================
    # ATS PROGRESS BAR
    # ============================================================

    st.markdown("### Overall Resume Compatibility")

    st.progress(
        ats_score / 100
    )

    st.write(f"### {verdict}")

    st.caption(verdict_msg)

    st.write("")

    # ============================================================
    # SUMMARY CARDS
    # ============================================================

    c1, c2 = st.columns(2)

    with c1:

        st.info(
            f"""
### Resume Strength

**{strength}**

This score is calculated using:

- Exact Skill Matching
- Semantic Skill Matching
- Missing Skills
            """
        )

    with c2:

        st.success(
            f"""
### Recruiter Readiness

**{readiness}**

A higher readiness indicates stronger
alignment with the job description.
            """
        )


# ============================================================
# MATCHED SKILLS
# ============================================================

section("✅ Matched Technical Skills")

if matched_exact:

    show_skill_badges(
        matched_exact,
        color="blue"
    )

else:

    st.warning(
        "No exact skill matches found."
    )


# ============================================================
# SEMANTIC MATCHES
# ============================================================

section("🔁 Semantic Skill Matches")

if semantic_matches:

    semantic_df = []

    for jd_skill, (resume_skill, score) in sorted(
        semantic_matches.items()
    ):

        semantic_df.append({

            "Job Skill": jd_skill.title(),

            "Resume Skill": resume_skill.title(),

            "Similarity": f"{score:.2f}",

            "Confidence": confidence_label(score)

        })

    semantic_df = pd.DataFrame(
        semantic_df
    )
    st.dataframe(
    semantic_df,
    use_container_width=True,
    hide_index=True
)

st.info(
    """
**Note**

These are semantic matches identified using the Word2Vec model.

They indicate related skills, **not exact keyword matches**.
"""
)
else:
    st.success(
        "No semantic substitutions were required."
    )


# ============================================================
# MISSING SKILLS
# ============================================================

section("❌ Missing Skills")

if true_gap:

    for skill in sorted(true_gap):

    priority = skill_priority(skill)

    if priority == "High":
        icon = "🔴"

    elif priority == "Medium":
        icon = "🟡"

    else:
        icon = "🟢"

    st.error(f"{icon} {skill.title()}")

    st.caption(
        recommendation(skill)
    )

    st.write("")

    rows = []

    for skill in sorted(true_gap):

        rows.append({

            "Missing Skill": skill.title(),

            "Priority": skill_priority(skill),

            "Recommendation": recommendation(skill)

        })

    missing_df = pd.DataFrame(rows)

    st.dataframe(

        missing_df,

        use_container_width=True,

        hide_index=True

    )

else:

    st.success(

        "Excellent! No important skills are missing."

    )


# ============================================================
# CHARTS
# ============================================================

section("📊 Skill Analytics")

chart1, chart2 = st.columns(2)

with chart1:

    df = metrics_dataframe(

        matched_exact,

        semantic_matches,

        true_gap

    )

    st.plotly_chart(

        ats_chart(df),

        use_container_width=True

    )

with chart2:

    st.plotly_chart(

        skill_donut(

            matched_exact,

            semantic_matches,

            true_gap

        ),

        use_container_width=True

    )
    # ============================================================
# CATEGORY WISE SKILLS
# ============================================================

section("📂 Skill Category Breakdown")

category_report = category_breakdown(resume_skills)

category_rows = []

for category, skills in category_report.items():

    if len(skills):

        status = "✅"

        skill_text = ", ".join(
            sorted([s.title() for s in skills])
        )

    else:

        status = "❌"

        skill_text = "No skills detected"

    category_rows.append({

        "Category": category,

        "Status": status,

        "Skills": skill_text

    })

category_df = pd.DataFrame(category_rows)

st.dataframe(

    category_df,

    use_container_width=True,

    hide_index=True

)

# ============================================================
# RECRUITER SUMMARY
# ============================================================
def recruiter_summary(
    ats_score,
    matched_exact,
    semantic_matches,
    missing
):

    summary = []

    summary.append(
        "Overall Assessment"
    )

    if ats_score >= 80:

        summary.append(
            "The resume demonstrates strong alignment with the target job description."
        )

    elif ats_score >= 60:

        summary.append(
            "The resume has a solid technical foundation and aligns with many of the required skills."
        )

    else:

        summary.append(
            "The resume currently lacks several important skills required for this role."
        )

    if matched_exact:

        summary.append(
            f"Exact matches: {len(matched_exact)} required technical skills."
        )

    if semantic_matches:

        summary.append(
            f"{len(semantic_matches)} additional related skills were identified through semantic matching."
        )

    if missing:

        summary.append(
            f"The resume is missing {len(missing)} important skills that should be added if applicable."
        )

    summary.append(
        "Adding relevant projects, certifications, quantified achievements, and missing technical skills can improve ATS compatibility."
    )

    return " ".join(summary)

# ============================================================
# IMPROVEMENT RECOMMENDATIONS
# ============================================================

section("💡 Personalized Recommendations")

tips = improvement_tips(true_gap)

if tips:

    for tip in tips:

        st.markdown(f"✅ {tip}")

else:

    st.success(

        "Your resume already aligns very well with this job description."

    )

# ============================================================
# SKILL GAP CLUSTERS
# ============================================================

if len(true_gap):

    section("📚 Suggested Skills to Learn")
    clusters = cluster_missing_skills(

        true_gap,

        cluster_count

    )

    for group, skills in clusters.items():

        with st.expander(group):

            for skill in sorted(skills):

                st.write("•", skill.title())


# ============================================================
# RESUME IMPROVEMENT CHECKLIST
# ============================================================

section("✅ Resume Improvement Checklist")

checklist = []

if "machine learning" in true_gap:
    checklist.append("Add at least one Machine Learning project.")

if "data engineering" in true_gap:
    checklist.append("Mention ETL or Data Pipeline experience.")

if ats_score < 80:
    checklist.append("Include more job-specific technical keywords.")

checklist.append("Quantify project achievements where possible.")
checklist.append("Keep GitHub and LinkedIn profiles updated.")

for item in checklist:
    st.markdown(f"✅ {item}")

# ============================================================
# DOWNLOAD REPORT
# ============================================================

section("📥 Export ATS Report")

report = pd.DataFrame({

    "Matched Skills":[

        ", ".join(sorted(matched_exact))

    ],

    "Semantic Matches":[

        ", ".join(

            f"{k}->{v[0]}"

            for k, v in semantic_matches.items()

        )

    ],

    "Missing Skills":[

        ", ".join(sorted(true_gap))

    ],

    "ATS Score":[

        ats_score

    ],

    "Resume Strength":[

        strength

    ],

    "Recruiter Readiness":[

        readiness

    ]

})

csv = report.to_csv(index=False).encode("utf-8")

st.download_button(

    "⬇ Download ATS Report (CSV)",

    csv,

    "SkillSync_Report.csv",

    "text/csv"

)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(

"""
<div style='text-align:center;color:gray;'>

### 🚀 SkillSync AI

Professional Resume Skill Gap Analyzer

Built with

**Python • Streamlit • spaCy • Word2Vec • KMeans • Plotly**

</div>
""",

unsafe_allow_html=True
)

    
