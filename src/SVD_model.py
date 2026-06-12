"""
svd_model.py
------------
Trains an SVD collaborative-filtering model using the Surprise library.
Exposes a predict() method and a bulk candidate scorer.
"""

import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy


class SVDRecommender:
    """Thin wrapper around Surprise SVD."""

    def __init__(self, n_factors: int = 50, n_epochs: int = 20, lr_all: float = 0.005, reg_all: float = 0.02):
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=42,
        )
        self.trainset = None
        self.testset = None

    def fit(self, ratings_df: pd.DataFrame, test_size: float = 0.2) -> dict:
        reader = Reader(rating_scale=(1, 5))

        data = Dataset.load_from_df(
            ratings_df[["user_id", "movie_id", "rating"]],
            reader,
         )

        self.trainset, self.testset = train_test_split(
        data,
        test_size=test_size,
        random_state=42,
        )

        self.model.fit(self.trainset)

        predictions = self.model.test(self.testset)

        rmse = float(accuracy.rmse(predictions, verbose=False))
        mae = float(accuracy.mae(predictions, verbose=False))

        return {
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
        }
       
        rmse = float(accuracy.rmse(predictions, verbose=False))
        mae = float(accuracy.mae(predictions, verbose=False))

        return {
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
        }

    def predict(self, user_id: int, movie_id: int) -> float:
        """Return predicted rating for a single (user, movie) pair."""
        return self.model.predict(user_id, movie_id).est

    def score_candidates(self, user_id: int, candidate_ids: list[int]) -> pd.Series:
        """
        Return a Series (movie_id → predicted_rating) for all candidates.
        Normalises scores to [0, 1] for easy blending downstream.
        """
        raw = {mid: self.predict(user_id, mid) for mid in candidate_ids}
        s = pd.Series(raw)
        # min-max normalise
        if s.max() > s.min():
            s = (s - s.min()) / (s.max() - s.min())
        return s

    def get_candidates(self, user_id: int, ratings_df: pd.DataFrame) -> list[int]:
        """Return movie IDs the user has NOT yet rated."""
        seen = set(ratings_df[ratings_df["user_id"] == user_id]["movie_id"])
        all_movies = set(ratings_df["movie_id"].unique())
        return list(all_movies - seen)