import re
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from docx import Document


st.set_page_config(
    page_title="AI Resume Screening & Job Recommendation",
    page_icon="🤖",
    layout="wide"
)

SKILLS = [
    "python", "java", "c++", "c#", "javascript", "typescript",
    "html", "css", "react", "angular", "node.js", "node",
    "django", "flask", "fastapi",
    "sql", "mysql", "postgresql", "mongodb",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "data analysis", "data visualization", "statistics",
    "excel", "power bi", "tableau",
    "git", "github", "docker", "aws", "azure", "gcp",
    "rest api", "api", "spark", "hadoop",
    "communication", "leadership", "problem solving"
]


@st.cache_data
def load_jobs():
    df = pd.read_csv("job_dataset.csv")
    df = df.fillna("")
    df["combined_text"] = (
        df["job_role"].astype(str) + " " +
        df["required_skills"].astype(str) + " " +
        df["qualification"].astype(str) + " " +
        df["experience"].astype(str) + " " +
        df["technologies"].astype(str) + " " +
        df["job_description"].astype(str) + " " +
        df["keywords"].astype(str)
    )
    return df


def extract_pdf_text(file_bytes):
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def extract_docx_text(file_bytes):
    document = Document(BytesIO(file_bytes))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    table_text = []
    for table in document.tables:
        for row in table.rows:
            table_text.append(" ".join(cell.text for cell in row.cells))
    return "\n".join(paragraphs + table_text)


def extract_text(uploaded_file):
    data = uploaded_file.getvalue()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return extract_pdf_text(data)
    if name.endswith(".docx"):
        return extract_docx_text(data)

    raise ValueError("Only PDF and DOCX files are supported.")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\- ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_skills(text):
    text_lower = text.lower()
    found = []

    # Check longer phrases first to avoid partial duplicates.
    for skill in sorted(SKILLS, key=len, reverse=True):
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, text_lower):
            found.append(skill)

    # Preserve the readable order of skills in the dictionary.
    return sorted(set(found), key=lambda x: SKILLS.index(x))


def get_experience_years(text):
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)",
        r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return None


def calculate_recommendations(resume_text, jobs):
    resume_clean = clean_text(resume_text)
    job_texts = jobs["combined_text"].tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform([resume_clean] + job_texts)
    resume_vector = matrix[0:1]
    job_vectors = matrix[1:]

    similarities = cosine_similarity(resume_vector, job_vectors)[0]
    scores = np.clip(similarities * 100, 0, 100)

    results = jobs.copy()
    results["match_score"] = scores
    results = results.sort_values("match_score", ascending=False).reset_index(drop=True)

    return results


def skill_overlap(resume_skills, required_skills_text):
    required = extract_skills(required_skills_text)
    if not required:
        return 0, []

    overlap = sorted(
        set(resume_skills).intersection(required),
        key=lambda x: required.index(x)
    )
    percentage = (len(overlap) / len(required)) * 100
    return percentage, overlap


# ---------------------- UI ----------------------

st.title("🤖 AI Resume Screening & Job Recommendation System")
st.caption("NLP-based resume analysis using TF-IDF and cosine similarity")

st.markdown(
    """
    Upload a resume in **PDF or DOCX** format. The system extracts the text,
    detects technical skills and keywords, compares the resume with job
    descriptions, and recommends suitable job roles.
    """
)

try:
    jobs = load_jobs()
except FileNotFoundError:
    st.error("job_dataset.csv was not found. Keep it in the same folder as app.py.")
    st.stop()

uploaded_file = st.file_uploader(
    "📄 Upload your resume",
    type=["pdf", "docx"],
    help="Supported formats: PDF and DOCX"
)

analyze = st.button("🔍 Analyze Resume", type="primary", use_container_width=True)

if uploaded_file and analyze:
    try:
        with st.spinner("Analyzing your resume..."):
            raw_text = extract_text(uploaded_file)

        if not raw_text.strip():
            st.error(
                "No readable text was found. If this is a scanned/image-only PDF, "
                "please upload a text-based PDF or DOCX."
            )
            st.stop()

        cleaned = clean_text(raw_text)
        resume_skills = extract_skills(raw_text)
        experience_years = get_experience_years(raw_text)
        results = calculate_recommendations(raw_text, jobs)

        st.success("Resume screening completed successfully.")

        # Resume overview
        st.subheader("📋 Resume Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Words Extracted", len(cleaned.split()))

        with col2:
            st.metric("Skills Detected", len(resume_skills))

        with col3:
            exp_label = f"{experience_years:g} yrs" if experience_years is not None else "Not detected"
            st.metric("Experience", exp_label)

        if resume_skills:
            st.markdown("### 🛠️ Detected Skills")
            st.write(", ".join(skill.title() for skill in resume_skills))
        else:
            st.info("No skills from the current skill dictionary were detected.")

        # Top recommendation
        top = results.iloc[0]
        top_required = str(top["required_skills"])
        top_skill_score, top_overlap = skill_overlap(resume_skills, top_required)

        st.subheader("🎯 Top Job Recommendation")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Recommended Role", str(top["job_role"]))

        with c2:
            st.metric("Match Score", f"{top['match_score']:.1f}%")

        with c3:
            st.metric("Skill Overlap", f"{top_skill_score:.1f}%")

        if top_overlap:
            st.caption(
                "Matching skills: " +
                ", ".join(skill.title() for skill in top_overlap)
            )

        st.markdown("### 🏆 Recommended Job Roles")

        display = results.head(5)[
            ["job_role", "match_score", "required_skills", "technologies"]
        ].copy()
        display["match_score"] = display["match_score"].map(lambda x: f"{x:.1f}%")
        display.columns = ["Job Role", "Match Score", "Required Skills", "Technologies"]

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        # Detailed cards
        st.subheader("📌 Job Details")

        for _, row in results.head(5).iterrows():
            with st.expander(f"{row['job_role']} — {row['match_score']:.1f}% match"):
                st.write(f"**Required Skills:** {row['required_skills']}")
                st.write(f"**Qualification:** {row['qualification']}")
                st.write(f"**Experience:** {row['experience']}")
                st.write(f"**Technologies:** {row['technologies']}")
                st.write(f"**Keywords:** {row['keywords']}")
                st.write(f"**Job Description:** {row['job_description']}")

        # Extracted text
        with st.expander("📄 View extracted resume text"):
            st.text(raw_text[:15000])

    except Exception as e:
        st.error(f"Could not analyze the resume: {e}")

else:
    st.info("Upload a resume and click **Analyze Resume** to begin.")

st.divider()

st.caption(
    "Project technique: NLP + TF-IDF text vectorization + cosine similarity. "
    "This application provides similarity-based recommendations and is not a "
    "substitute for human recruitment decisions."
)
