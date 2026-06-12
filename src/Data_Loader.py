"""
data_loader.py
--------------
Downloads and parses the MovieLens 100K dataset.
Returns clean pandas DataFrames ready for modelling.
"""

import os
import zipfile
import requests
import pandas as pd

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def download_movielens(data_dir: str = DATA_DIR) -> None:
    """Download and unzip MovieLens 100K if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ml-100k.zip")
    extracted_dir = os.path.join(data_dir, "ml-100k")

    if os.path.isdir(extracted_dir):
        print("Dataset already downloaded.")
        return

    print("Downloading MovieLens 100K (~5 MB)...")
    resp = requests.get(DATA_URL, timeout=60)
    resp.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(resp.content)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    os.remove(zip_path)
    print("Download complete.")


def load_ratings(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load ratings. Columns: user_id, movie_id, rating, timestamp."""
    path = os.path.join(data_dir, "ml-100k", "u.data")
    df = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    return df


def load_movies(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load movie metadata. Columns: movie_id, title, genres (list)."""
    path = os.path.join(data_dir, "ml-100k", "u.item")
    genre_cols = [
        "unknown", "Action", "Adventure", "Animation", "Children",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western",
    ]
    cols = ["movie_id", "title", "release_date", "video_date", "imdb_url"] + genre_cols
    df = pd.read_csv(path, sep="|", names=cols, encoding="latin-1")

    # Build a human-readable genres list per movie
    df["genres"] = df[genre_cols].apply(
        lambda row: [g for g, v in zip(genre_cols, row) if v == 1], axis=1
    )
    return df[["movie_id", "title", "genres"]]