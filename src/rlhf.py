import os
import pandas as pd

FEEDBACK_FILE = "data/feedback.csv"


def load_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        return pd.DataFrame(
            columns=["user_id", "movie_id", "reward"]
        )

    return pd.read_csv(FEEDBACK_FILE)


def add_feedback(user_id, movie_id, reward):
    df = load_feedback()

    new_row = pd.DataFrame([
        {
            "user_id": user_id,
            "movie_id": movie_id,
            "reward": reward,
        }
    ])

    df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(FEEDBACK_FILE, index=False)


def learn_alpha(user_id):
    feedback = load_feedback()

    user_fb = feedback[
        feedback["user_id"] == user_id
    ]

    if len(user_fb) == 0:
        return 0.6

    avg_reward = user_fb["reward"].mean()

    alpha = 0.6 + (0.2 * avg_reward)

    alpha = max(0.3, min(0.9, alpha))

    return round(alpha, 2)


def feedback_summary(user_id):
    feedback = load_feedback()

    user_fb = feedback[
        feedback["user_id"] == user_id
    ]

    if len(user_fb) == 0:
        return "No feedback yet."

    avg_reward = round(user_fb["reward"].mean(), 3)

    return (
        f"Feedback entries: {len(user_fb)} | "
        f"Average reward: {avg_reward}"
    )