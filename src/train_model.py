import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("THIS IS THE NEW TRAIN MODEL")

# ----------------------------------------------------
# Load preprocessed dataset
# ----------------------------------------------------
df = pd.read_csv("data/cleaned_github_repos.csv")

print("Dataset Shape:", df.shape)

# ----------------------------------------------------
# Select features
# We DO NOT use:
# - stars (target label)
# - forks
# - watchers
# ----------------------------------------------------
features = [
    "issues",
    "size",
    "language",
    "repo_age_days",
    "days_since_update",
    "issue_density",
    "has_issues",
    "has_wiki"
]

print(df.columns.tolist())
print(features)

X = df[features]
y = df["success"]

# ----------------------------------------------------
# Split dataset into training and testing
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ----------------------------------------------------
# Scale data
# (Needed for Logistic Regression)
# ----------------------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====================================================
# Logistic Regression
# ====================================================
print("\n========== Logistic Regression ==========")

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)

print("Accuracy:", round(accuracy_score(y_test, lr_pred), 4))

print("\nClassification Report:")
print(classification_report(y_test, lr_pred))

# ====================================================
# Random Forest
# ====================================================
print("\n========== Random Forest ==========")

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Random Forest uses unscaled data
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, rf_pred), 4))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# ====================================================
# Confusion Matrix
# ====================================================
print("\nConfusion Matrix:")

print(confusion_matrix(y_test, rf_pred))

# ====================================================
# Feature Importance
# ====================================================
importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

# ====================================================
# Save Model
# ====================================================


# Save trained Random Forest model
joblib.dump(rf, "models/github_success_model.pkl")

# Save scaler (used by Logistic Regression)
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel saved successfully!")