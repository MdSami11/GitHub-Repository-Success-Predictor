import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("data/github_repos.csv")

print("Initial Shape:", df.shape)

# -------------------------
# Remove missing values
# -------------------------
df = df.dropna()

print("After Cleaning:", df.shape)

# -------------------------
# Convert dates
# -------------------------
df["created_at"] = pd.to_datetime(df["created_at"])
df["updated_at"] = pd.to_datetime(df["updated_at"])

# -------------------------
# Feature Engineering
# -------------------------

# Current date
today = pd.Timestamp.now(tz="UTC")

# Repository age (days)
df["repo_age_days"] = (today - df["created_at"]).dt.days

# Days since last update
df["days_since_update"] = (today - df["updated_at"]).dt.days

# Issue density
df["issue_density"] = df["issues"] / (df["size"] + 1)


# Encode language into numbers
encoder = LabelEncoder()
df["language"] = encoder.fit_transform(df["language"])

# Save the trained LabelEncoder
joblib.dump(encoder, "models/language_encoder.pkl")

# -------------------------
# Create Target Label
# -------------------------
# Repository is considered successful if it has >= 500 stars

threshold = 500

df["success"] = (df["stars"] >= threshold).astype(int)

print("\nTarget Distribution:")
print(df["success"].value_counts())

# -------------------------
# Balance Dataset
# -------------------------
df0 = df[df["success"] == 0]
df1 = df[df["success"] == 1]

min_size = min(len(df0), len(df1))

df0 = df0.sample(min_size, random_state=42)
df1 = df1.sample(min_size, random_state=42)

df = pd.concat([df0, df1])

# Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced Dataset:")
print(df["success"].value_counts())

# -------------------------
# Save processed dataset
# -------------------------
df.to_csv(
    "data/cleaned_github_repos.csv",
    index=False
)

print("\nPreprocessing Completed Successfully!")