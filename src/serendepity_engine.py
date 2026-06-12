"""
serendipity_engine.py
---------------------
NOVELTY: Serendipity Engine
===========================
Most recommenders just maximise predicted rating (relevance).
We add a *surprise* dimension:

    serendipity_score = α * relevance + (1 - α) * surprise

Where:
  - relevance  = normalised SVD predicted rating  (0..1)
  - surprise   = how far the item is from the user's genre profile (0..1)
  - α          = tunable knob (default 0.6) — higher = safer, lower = wilder

The genre-distance surprise score works like this:
  1. Build the user's genre vector: average of genre one-hot vectors for
     their highly-rated movies (rating >= 4).
  2. For each candidate movie, compute cosine distance from that profile.
  3. High distance = high surprise.

This avoids the "filter bubble" where you only ever get more of what
you already watch, while still ensuring the items are predicted to be
good (relevance gate).
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


GENRE_LIST = [
    "unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]


def build_genre_matrix(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a (n_movies × n_genres) binary DataFrame indexed by movie_id.
    """
    def genres_to_vector(genre_list):
        return [1 if g in genre_list else 0 for g in GENRE_LIST]

    vectors = movies_df["genres"].apply(genres_to_vector).tolist()
    matrix = pd.DataFrame(vectors, index=movies_df["movie_id"], columns=GENRE_LIST)
    return matrix.astype(float)


def build_user_genre_profile(
    user_id: int,
    ratings_df: pd.DataFrame,
    genre_matrix: pd.DataFrame,
    min_rating: float = 4.0,
) -> np.ndarray:
    """
    Compute the user's genre taste vector by averaging genre vectors
    of their highly-rated movies.
    Falls back to all rated movies if no high ratings exist.
    """
    user_ratings = ratings_df[ratings_df["user_id"] == user_id]
    liked = user_ratings[user_ratings["rating"] >= min_rating]["movie_id"]
    if len(liked) == 0:
        liked = user_ratings["movie_id"]  # fallback

    # Keep only movies present in the genre matrix
    liked = liked[liked.isin(genre_matrix.index)]
    if len(liked) == 0:
        return np.zeros(len(GENRE_LIST))

    profile = genre_matrix.loc[liked].mean(axis=0).values
    return profile


def compute_surprise_scores(
    candidate_ids: list[int],
    user_profile: np.ndarray,
    genre_matrix: pd.DataFrame,
) -> pd.Series:
    """
    Surprise score = 1 - cosine_similarity(item_genre_vector, user_profile).
    Items far from the user's taste score high on surprise.
    Normalised to [0, 1].
    """
    # Filter candidates that have a genre vector
    valid = [m for m in candidate_ids if m in genre_matrix.index]
    if not valid:
        return pd.Series(dtype=float)

    item_vectors = genre_matrix.loc[valid].values
    profile_vec = user_profile.reshape(1, -1)

    # Guard: zero-norm profile means we have no taste signal
    if np.linalg.norm(profile_vec) == 0:
        return pd.Series(0.5, index=valid)

    sim = cosine_similarity(item_vectors, profile_vec).flatten()
    surprise = 1.0 - sim  # high similarity → low surprise

    # Normalise
    s = pd.Series(surprise, index=valid)
    if s.max() > s.min():
        s = (s - s.min()) / (s.max() - s.min())
    return s


def serendipity_score(
    relevance: pd.Series,
    surprise: pd.Series,
    alpha: float = 0.6,
) -> pd.Series:
    """
    Blend relevance and surprise into a single serendipity score.

    Args:
        relevance : Series (movie_id → normalised predicted rating)
        surprise  : Series (movie_id → normalised genre distance)
        alpha     : weight for relevance (1-alpha goes to surprise)

    Returns:
        Series (movie_id → serendipity score), sorted descending.
    """
    common = relevance.index.intersection(surprise.index)
    score = alpha * relevance[common] + (1 - alpha) * surprise[common]
    return score.sort_values(ascending=False)


def recommend(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    svd_recommender,
    genre_matrix: pd.DataFrame,
    alpha: float = 0.6,
    top_n: int = 10,
    min_relevance: float = 0.3,
) -> pd.DataFrame:
    """
    Full serendipity recommendation pipeline.

    1. Get candidate movies (unseen by user).
    2. Score relevance via SVD.
    3. Build user genre profile and score surprise.
    4. Blend and filter.
    5. Return a DataFrame with full metadata.
    """
    candidates = svd_recommender.get_candidates(user_id, ratings_df)

    # Step 1: relevance
    relevance = svd_recommender.score_candidates(user_id, candidates)

    # Step 2: filter out low-relevance items first (relevance gate)
    # This ensures we never recommend irrelevant surprises
    relevance = relevance[relevance >= min_relevance]

    # Step 3: surprise
    user_profile = build_user_genre_profile(user_id, ratings_df, genre_matrix)
    surprise = compute_surprise_scores(relevance.index.tolist(), user_profile, genre_matrix)

    # Step 4: blend
    scores = serendipity_score(relevance, surprise, alpha=alpha)

    # Step 5: enrich with movie metadata
    top_ids = scores.head(top_n).index.tolist()
    result = movies_df[movies_df["movie_id"].isin(top_ids)].copy()
    result["serendipity_score"] = result["movie_id"].map(scores)
    result["relevance_score"]   = result["movie_id"].map(relevance)
    result["surprise_score"]    = result["movie_id"].map(surprise)
    result["predicted_rating"]  = result["movie_id"].apply(
        lambda mid: round(svd_recommender.predict(user_id, mid), 2)
    )
    result = result.sort_values("serendipity_score", ascending=False)
    result["genres"] = result["genres"].apply(lambda g: ", ".join(g) if g else "—")
    result = result[["movie_id", "title", "genres", "predicted_rating",
                     "relevance_score", "surprise_score", "serendipity_score"]]
    result = result.round(3)
    return result.reset_index(drop=True)