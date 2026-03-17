"""
scripts/generate_sample_data.py
────────────────────────────────
Generates realistic sample data that mirrors the Netflix Prize + TMDB datasets.
Run this ONCE before running bronze_ingestion.py.

Creates:
  data/raw/ratings_1.csv        – 200 user ratings  (CSV format)
  data/raw/movies.json          – 30 movies          (JSON format)
  data/raw/users/users.parquet  – 50 users           (Parquet format)
  data/raw/ratings_bad.csv      – intentionally corrupt file (tests quarantine)
"""

import json
import os
import random
from datetime import datetime, timedelta

import pandas as pd

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(BASE, "data", "raw")
os.makedirs(RAW, exist_ok=True)
os.makedirs(os.path.join(RAW, "users"), exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. RATINGS  →  CSV   (mirrors Netflix Prize combined_data format)
# ─────────────────────────────────────────────────────────────────────────────
def make_ratings_csv(path: str, n: int = 200) -> None:
    base_date = datetime(2024, 1, 1)
    rows = []
    for _ in range(n):
        rows.append({
            "user_id":  random.randint(1, 50),
            "movie_id": random.randint(1, 30),
            "rating":   round(random.uniform(1.0, 5.0) * 2) / 2,   # 0.5 steps
            "rated_at": (base_date + timedelta(
                days=random.randint(0, 90),
                hours=random.randint(0, 23),
            )).strftime("%Y-%m-%d %H:%M:%S"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  Created {path}  ({len(df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. MOVIES  →  JSON   (mirrors TMDB movies_metadata)
# ─────────────────────────────────────────────────────────────────────────────
GENRES   = ["Action", "Comedy", "Drama", "Thriller", "Romance",
            "Sci-Fi", "Horror", "Animation", "Documentary"]
LANGS    = ["en", "fr", "es", "de", "ja", "ko"]
COUNTRIES = ["US", "GB", "FR", "DE", "JP", "KR", "IN"]

TITLES = [
    "The Lost City", "Midnight Echo", "Broken Wings", "Storm Rising",
    "The Last Signal", "Pale Fire", "Iron Tide", "Wandering Stars",
    "The Deep Blue", "City of Shadows", "The Forgotten Road", "Silver Lining",
    "Dark Horizon", "The Last Train", "Echo Chamber", "Fading Light",
    "The Open Sea", "Ghost Protocol", "Red Canyon", "Winter's End",
    "The Inheritance", "Running Man", "Blue Lagoon", "Night Watch",
    "The Outsider", "Final Cut", "Rising Sun", "Cold Harbor",
    "The Long Game", "Burning Bridges",
]

def make_movies_json(path: str) -> None:
    movies = []
    for i, title in enumerate(TITLES, start=1):
        genre_count = random.randint(1, 3)
        movies.append({
            "movie_id": i,
            "title":    title,
            "genres":   "|".join(random.sample(GENRES, genre_count)),
            "year":     random.randint(1990, 2024),
            "language": random.choice(LANGS),
            "country":  random.choice(COUNTRIES),
        })
    with open(path, "w") as f:
        json.dump(movies, f, indent=2)
    print(f"  Created {path}  ({len(movies)} records)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. USERS  →  Parquet   (synthetic subscription data)
# ─────────────────────────────────────────────────────────────────────────────
PLANS     = ["basic", "standard", "premium"]

def make_users_parquet(path: str, n: int = 50) -> None:
    base_date = datetime(2022, 1, 1)
    rows = []
    for uid in range(1, n + 1):
        rows.append({
            "user_id":     uid,
            "signup_date": (base_date + timedelta(days=random.randint(0, 700))
                            ).strftime("%Y-%m-%d"),
            "country":     random.choice(COUNTRIES),
            "plan":        random.choice(PLANS),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    print(f"  Created {path}  ({len(df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. BAD FILE  →  CSV with corrupt content  (tests Q5 quarantine logic)
# ─────────────────────────────────────────────────────────────────────────────
def make_bad_csv(path: str) -> None:
    content = (
        "user_id,movie_id,rating,rated_at\n"
        "CORRUPT_LINE%%%###\n"
        "not,a,valid,row,extra_col\n"
        "\x00\x01\x02binary garbage\n"
    )
    with open(path, "w") as f:
        f.write(content)
    print(f"  Created {path}  (intentionally corrupt — tests quarantine)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nGenerating sample data...")
    make_ratings_csv(os.path.join(RAW, "ratings_1.csv"))
    make_movies_json(os.path.join(RAW, "movies.json"))
    make_users_parquet(os.path.join(RAW, "users", "users.parquet"))
    make_bad_csv(os.path.join(RAW, "ratings_bad.csv"))
    print("\nDone! Files are in data/raw/")
    print("Run scripts/run_bronze.py next to ingest them into Bronze.")
