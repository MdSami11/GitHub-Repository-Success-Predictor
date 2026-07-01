# 🚀 GitHub Repository Success Predictor

A Machine Learning project that predicts whether a GitHub repository is likely to become successful based on repository activity, metadata, and engineering signals.

Built end-to-end: from data collection using GitHub API → feature engineering → model training → deployment using Streamlit.

---

## 🎯 Problem Statement

GitHub has millions of repositories, but only a small fraction become popular or successful.

This project aims to predict repository success early using measurable signals like:
- activity
- size
- issues
- update frequency
- metadata patterns

---

## 🧠 Solution Overview

I built a full ML pipeline that:
1. Collects real GitHub repository data using GitHub API
2. Cleans and preprocesses the dataset
3. Engineers meaningful features
4. Trains and compares ML models
5. Deploys the best model using a web interface

---

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- GitHub API
- Joblib

---

## 📊 Features Used

- Open Issues
- Repository Size
- Programming Language (encoded)
- Repository Age
- Days Since Last Update
- Issue Density
- Has Issues Enabled
- Has Wiki Enabled

---

## 🤖 Machine Learning Models

- Logistic Regression (baseline)
- Random Forest Classifier (final model)

### Performance
- Logistic Regression: ~83%
- Random Forest: ~90%

Final model selected: **Random Forest Classifier**

---

## 🧪 How It Works

1. User enters repository details in web app
2. Input is preprocessed (same format as training data)
3. Trained Random Forest model predicts success
4. Output is displayed with confidence score

---

## 🌐 Live Demo (Optional)

Run locally:

```bash
streamlit run app.py