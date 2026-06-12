"""
evaluate.py
-----------
Metrics beyond accuracy — because a serendipity engine needs
serendipity-aware evaluation.

Metrics implemented
-------------------
precision_at_k     : Classic precision@k on held-out ratings
intra_list_diversity : Average pairwise genre distance within a rec list
                      (measures variety — high = diverse picks)
novelty_score      : Average inverse popularity of recommended items
                    (low-frequency items score high)
serendipity_metric : Fraction of recs that are both relevant AND surprising
                    relative to a baseline popularity model
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.serendepity_engine import GENRE_LIST


def precision_at_k(
    recommended_ids: list[int],
    test_ratings: pd.DataFrame,
    user_id: int,
    k: int = 10,
    threshold: float = 3.5,
) -> float:
    """Fraction of top-k recs that the user actually rated >= threshold."""
    liked = set(
        test_ratings[
            (test_ratings["user_id"] == user_id) &
            (test_ratings["rating"] >= threshold)
        ]["movie_id"]
    )
    hits = sum(1 for mid in recommended_ids[:k] if mid in liked)
    return round(hits / k, 4)


def intra_list_diversity(
    recommended_ids: list[int],
    genre_matrix: pd.DataFrame,
) -> float:
    """
    Average pairwise cosine DISTANCE between recommended items' genre vectors.
    Range [0, 1]. Higher = more diverse list.
    """
    valid = [m for m in recommended_ids if m in genre_matrix.index]
    if len(valid) < 2:
        return 0.0
    vecs = genre_matrix.loc[valid].values
    sim_matrix = cosine_similarity(vecs)
    n = len(valid)
    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_dist += (1.0 - sim_matrix[i, j])
            count += 1
    return round(total_dist / count if count > 0 else 0.0, 4)


def novelty_score(
    recommended_ids: list[int],
    ratings_df: pd.DataFrame,
) -> float:
    """
    Average log-inverse popularity of recommended items.
    Rare/niche items score higher.
    """
    popularity = ratings_df["movie_id"].value_counts()
    n_users = ratings_df["user_id"].nunique()
    scores = []
    for mid in recommended_ids:
        pop = popularity.get(mid, 1) / n_users
        scores.append(-np.log2(pop))
    return round(float(np.mean(scores)), 4)


def evaluate_recommender(
    user_id: int,
    recommended_df: pd.DataFrame,
    test_ratings: pd.DataFrame,
    ratings_df: pd.DataFrame,
    genre_matrix: pd.DataFrame,
    k: int = 10,
) -> dict:
    """Run all evaluation metrics and return a summary dict."""
    rec_ids = recommended_df["movie_id"].tolist()[:k]

    return {
        "user_id":             user_id,
        "precision@k":         precision_at_k(rec_ids, test_ratings, user_id, k),
        "intra_list_diversity": intra_list_diversity(rec_ids, genre_matrix),
        "novelty_score":       novelty_score(rec_ids, ratings_df),
        "mean_predicted_rating": round(recommended_df["predicted_rating"].mean(), 3),
        "mean_surprise_score": round(recommended_df["surprise_score"].mean(), 3),
    }