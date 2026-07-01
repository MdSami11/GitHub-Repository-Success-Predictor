import requests
import pandas as pd
import time

# -------------------------
# GitHub Search API
# -------------------------
BASE_URL = "https://api.github.com/search/repositories"

# Collect repositories from different popularity ranges
star_ranges = [
    "0..20",
    "21..100",
    "101..500",
    "501..2000",
    ">2000"
]

repos = []

# -------------------------
# Collect repositories
# -------------------------
for star_range in star_ranges:

    print(f"\nCollecting repositories with stars: {star_range}")

    url = (
        f"{BASE_URL}"
        f"?q=stars:{star_range}"
        f"&sort=stars"
        f"&order=desc"
        f"&per_page=100"
    )

    response = requests.get(url)

    if response.status_code != 200:
        print("Error:", response.status_code)
        continue

    data = response.json()["items"]

    for repo in data:

        repos.append({

            # Repository Info
            "name": repo["name"],
            "language": repo["language"],

            # Used only for creating the target label
            "stars": repo["stargazers_count"],

            # Repository Activity
            "issues": repo["open_issues_count"],

            # Repository Size
            "size": repo["size"],

            # Dates
            "created_at": repo["created_at"],
            "updated_at": repo["updated_at"],

            # Repository Settings
            "has_issues": repo["has_issues"],
            "has_wiki": repo["has_wiki"]

        })

    print(f"Collected {len(data)} repositories")

    # Avoid hitting GitHub rate limits
    time.sleep(2)

# -------------------------
# Save dataset
# -------------------------
df = pd.DataFrame(repos)

print("\nDataset Shape:", df.shape)

df.to_csv("data/github_repos.csv", index=False)

print("\nData saved successfully!")

print(df.columns.tolist())