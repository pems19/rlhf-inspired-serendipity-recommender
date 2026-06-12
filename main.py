from src.rlhf import (
    learn_alpha,
    add_feedback,
    feedback_summary,
)

from src.Data_Loader import (
    download_movielens,
    load_ratings,
    load_movies,
)

from src.SVD_model import SVDRecommender

from src.serendepity_engine import (
    build_genre_matrix,
    recommend,
)

from src.Evaluate import evaluate_recommender

import os
import json


def main():
    print("Loading MovieLens dataset...")
    download_movielens()

    ratings_df = load_ratings()
    movies_df = load_movies()

    print(f"Ratings loaded: {len(ratings_df)}")
    print(f"Movies loaded : {len(movies_df)}")

    print("\nTraining SVD model...")
    recommender = SVDRecommender()
    metrics = recommender.fit(ratings_df)

    print("\n===== MODEL PERFORMANCE =====")
    print(f"RMSE : {metrics['RMSE']}")
    print(f"MAE  : {metrics['MAE']}")

    print("\nBuilding genre matrix...")
    genre_matrix = build_genre_matrix(movies_df)

    user_id = 1

    print(f"\nGenerating recommendations for User {user_id}...")

    alpha = learn_alpha(user_id)

    print(f"\nAdaptive Alpha Learned From Feedback: {alpha}")
    print(feedback_summary(user_id))

    if alpha > 0.7:
        print("User prefers safer recommendations.")
    elif alpha < 0.5:
        print("User prefers exploratory recommendations.")
    else:
        print("User likes a balance of relevance and surprise.")

    recommendations = recommend(
        user_id=user_id,
        ratings_df=ratings_df,
        movies_df=movies_df,
        svd_recommender=recommender,
        genre_matrix=genre_matrix,
        alpha=alpha,
        top_n=10,
    )

    print("\n===== TOP 10 RECOMMENDATIONS =====")
    print(recommendations.to_string(index=False))

    # =========================
    # RLHF FEEDBACK COLLECTION
    # =========================

    print("\n===== FEEDBACK COLLECTION =====")
    print("Rate the first 3 recommendations")
    print("1   = Like")
    print("0.5 = Watch")
    print("0   = Ignore")
    print("-1  = Dislike")

    for _, row in recommendations.head(3).iterrows():

        print("\n----------------------------------")
        print(f"Movie : {row['title']}")
        print(f"Genres: {row['genres']}")

        while True:
            try:
                reward = float(
                    input("Feedback (1, 0.5, 0, -1): ")
                )

                if reward not in [1, 0.5, 0, -1]:
                    raise ValueError

                break

            except ValueError:
                print("Please enter 1, 0.5, 0, or -1")

        add_feedback(
            user_id,
            row["movie_id"],
            reward,
        )

    print("\nFeedback saved successfully!")

    # =========================
    # SAVE OUTPUTS
    # =========================

    os.makedirs("outputs", exist_ok=True)

    recommendations.to_csv(
        "outputs/recommendations.csv",
        index=False,
    )

    print("\nRecommendations saved to:")
    print("outputs/recommendations.csv")

    # =========================
    # EVALUATION
    # =========================

    print("\nEvaluating recommendations...")
    print(
        "NOTE: precision@k may be 0 because recommendations "
        "are generated from unseen movies while evaluation "
        "uses the complete ratings dataset."
    )

    evaluation = evaluate_recommender(
        user_id=user_id,
        recommended_df=recommendations,
        test_ratings=ratings_df,
        ratings_df=ratings_df,
        genre_matrix=genre_matrix,
        k=10,
    )

    print("\n===== EVALUATION =====")

    for metric, value in evaluation.items():
        print(f"{metric}: {value}")

    with open("outputs/metrics.json", "w") as f:
        json.dump(evaluation, f, indent=4)

    print("\nMetrics saved to:")
    print("outputs/metrics.json")

    # =========================
    # FINAL SUMMARY
    # =========================

    print("\n===== PROJECT COMPLETE =====")
    print(f"Final Alpha Used: {alpha}")

    if alpha > 0.7:
        print("User prefers safe/high-confidence recommendations.")
    elif alpha < 0.5:
        print("User prefers exploratory/surprising recommendations.")
    else:
        print("User likes a balance of relevance and surprise.")


if __name__ == "__main__":
    main()