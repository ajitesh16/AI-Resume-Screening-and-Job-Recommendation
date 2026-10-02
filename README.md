# 🤖 AI Resume Screening and Job Recommendation System

An NLP-based application that analyzes resumes and recommends suitable job roles by comparing resume content with job descriptions.

## 🚀 Live Demo

👉 **[Try the AI Resume Screening & Job Recommendation System](https://ai-resume-screening-and-job-recommendation.streamlit.app/)**

Upload a PDF or DOCX resume and get job recommendations based on skills and text similarity.

## Project Overview

Recruiters may need to screen a large number of resumes while candidate skills, qualifications and experience vary widely. This project automates the first stage of screening by extracting resume text, detecting relevant skills and comparing the resume with a collection of job descriptions.

The project follows the workflow specified in the project presentation:

**Resume Upload → Text Extraction → Text Preprocessing → Skill & Keyword Extraction → TF-IDF → Cosine Similarity → Match Score → Job Recommendations**

## Objectives

- Automatically analyze resumes
- Extract text, skills and qualifications
- Identify relevant keywords
- Compare resumes with job descriptions
- Recommend suitable job roles
- Display an interpretable similarity-based match score

## Features

- PDF resume upload
- DOCX resume upload
- Resume text extraction
- Basic NLP text cleaning
- Skill detection using a configurable skill dictionary
- TF-IDF vectorization
- Cosine similarity matching
- Top job recommendations
- Match percentage for each role
- Detected skill display
- Required skills, technologies and job description details
- Streamlit web interface

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- PyPDF2
- python-docx
- NLP / Text Similarity
- TF-IDF
- Cosine Similarity

## Dataset

The included `job_dataset.csv` contains sample job descriptions and structured job features for demonstration and evaluation.

Columns:

- `job_role`
- `required_skills`
- `qualification`
- `experience`
- `technologies`
- `job_description`
- `keywords`

The dataset can be expanded with additional real job descriptions for future versions.

## How the Matching Works

### 1. Resume Text Extraction

The application extracts readable text from a PDF or DOCX resume.

### 2. Text Preprocessing

Text is converted to lowercase, whitespace is normalized and unnecessary characters are removed.

### 3. Skill Extraction

The application checks the resume against a configurable technical/soft-skill dictionary.

### 4. TF-IDF

The resume and job descriptions are converted into TF-IDF vectors.

### 5. Cosine Similarity

Cosine similarity measures how similar the resume text is to each job description.

### 6. Recommendation

Job roles are sorted by similarity score and the top matches are displayed.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AI-Resume-Screening-and-Job-Recommendation.git
cd AI-Resume-Screening-and-Job-Recommendation
```

Create a virtual environment (recommended):

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

If the `streamlit` command is not recognized on Windows, use:

```bash
python -m streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## Project Structure

```text
AI-Resume-Screening-and-Job-Recommendation/
│
├── app.py
├── job_dataset.csv
├── requirements.txt
└── README.md
```

## Example Workflow

1. Open the Streamlit application.
2. Upload a text-based PDF or DOCX resume.
3. Click **Analyze Resume**.
4. Review extracted skills.
5. Review the top recommended job roles.
6. Compare match scores and job requirements.

## Evaluation

The application can be evaluated using test resumes representing different career profiles, such as:

- Python Developer
- Data Analyst
- Machine Learning Engineer
- Web Developer

For a formal evaluation, labeled resumes and job descriptions can be collected and metrics such as precision, recall and F1-score can be calculated for role classification/recommendation experiments.

## Limitations

- The current skill dictionary is manually configured.
- The included job dataset is a demonstration dataset and should be expanded for production use.
- Scanned image-only PDFs may not contain extractable text because OCR is not included in this version.
- Similarity scores indicate textual similarity and should not be treated as a definitive hiring decision.

## Future Scope

- OCR support for scanned resumes
- Larger real-world job-description dataset
- Advanced named-entity recognition using spaCy
- Skill-gap analysis
- Automated candidate ranking
- Personalized job recommendations
- Resume improvement suggestions
- Interview-question generation
- Job-portal integration
- AI career guidance chatbot

## Project Context

This project implements an intermediate AI/ML resume screening and job recommendation workflow using Python, NLP, machine learning techniques and Scikit-learn.
