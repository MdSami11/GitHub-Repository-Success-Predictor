import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Load trained Random Forest model
# --------------------------------------------------
model = joblib.load("models/github_success_model.pkl")

# Load the saved LabelEncoder
encoder = joblib.load("models/language_encoder.pkl")

# --------------------------------------------------
# Page Title
# --------------------------------------------------
st.title("🚀 GitHub Repository Success Predictor")

st.write(
    "Enter repository information below to predict whether "
    "the repository is likely to become successful."
)

# --------------------------------------------------
# User Inputs
# --------------------------------------------------

issues = st.number_input(
    "Open Issues",
    min_value=0,
    value=10
)

size = st.number_input(
    "Repository Size (KB)",
    min_value=1,
    value=500
)

language = st.selectbox(
    "Programming Language",
    [
        "Python",
        "Java",
        "JavaScript",
        "C++",
        "C",
        "Go",
        "Rust",
        "TypeScript"
    ]
)

repo_age_days = st.number_input(
    "Repository Age (Days)",
    min_value=1,
    value=365
)

days_since_update = st.number_input(
    "Days Since Last Update",
    min_value=0,
    value=30
)

issue_density = issues / (size + 1)

has_issues = st.selectbox(
    "Issues Enabled",
    ["Yes", "No"]
)

has_wiki = st.selectbox(
    "Wiki Enabled",
    ["Yes", "No"]
)


# Convert input language exactly as during training in numbers
language = encoder.transform([language])[0]

has_issues = 1 if has_issues == "Yes" else 0
has_wiki = 1 if has_wiki == "Yes" else 0

# --------------------------------------------------
# Predict Button
# --------------------------------------------------

if st.button("Predict"):

    # Create DataFrame in same order used during training
    input_data = pd.DataFrame([{
        "issues": issues,
        "size": size,
        "language": language,
        "repo_age_days": repo_age_days,
        "days_since_update": days_since_update,
        "issue_density": issue_density,
        "has_issues": has_issues,
        "has_wiki": has_wiki
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success("✅ This repository is likely to be successful.")
    else:
        st.error("❌ This repository is unlikely to be successful.")

    st.write(f"Confidence: {max(probability)*100:.2f}%")