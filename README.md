# Adaptive Serendipity-Aware Movie Recommender

An intelligent movie recommendation system that goes beyond traditional collaborative filtering by balancing **relevance** and **surprise**. The project uses SVD-based collaborative filtering, a custom serendipity engine, and an RLHF-inspired feedback loop to personalize recommendations over time.

---

## Problem Statement

Most recommendation systems optimize only for accuracy and relevance. While effective, they often create **filter bubbles**, repeatedly recommending content similar to what users have already consumed.

This project addresses that limitation by introducing **serendipity-aware recommendations** that surface movies users are likely to enjoy but may not have discovered on their own.

---

## Features

### Collaborative Filtering

* SVD-based recommendation model using the Surprise library
* Trained on the MovieLens 100K dataset
* Predicts user-movie ratings

### Serendipity Engine

Recommendations are ranked using:

**Serendipity Score = α × Relevance + (1 − α) × Surprise**

Where:

* **Relevance** = normalized SVD prediction score
* **Surprise** = genre-distance novelty score
* **α** = personalization parameter

### RLHF-Inspired Feedback Loop

Users provide explicit feedback on recommendations:

| Feedback | Reward |
| -------- | ------ |
| Like     | +1     |
| Watch    | +0.5   |
| Ignore   | 0      |
| Dislike  | -1     |

The system stores feedback and learns a personalized alpha value that dynamically adjusts the balance between relevance and surprise.

### Evaluation Metrics

* RMSE
* MAE
* Precision@K
* Novelty Score
* Intra-List Diversity

---

## System Architecture

```text
MovieLens Dataset
        │
        ▼
SVD Collaborative Filtering
        │
        ▼
Relevance Scores
        │
        ▼
Serendipity Engine
(Genre-Distance Novelty)
        │
        ▼
Serendipity Ranking
        │
        ▼
Recommendations
        │
        ▼
User Feedback
        │
        ▼
Adaptive Alpha Learning
(RLHF-Inspired)
        │
        └──────────────► Future Recommendations
```

---

## Dataset

MovieLens 100K Dataset

* 100,000 ratings
* 943 users
* 1,682 movies
* Ratings scale: 1–5

Dataset source:
https://grouplens.org/datasets/movielens/

---

## Example Recommendation Output

```text
movie_id   title                          predicted_rating
525        Big Sleep, The (1946)               4.44
661        High Noon (1952)                    4.40
484        Maltese Falcon, The (1941)          4.39
```

Example metrics:

```text
RMSE : 0.9348
MAE  : 0.7377
Novelty Score : 3.412
Intra-List Diversity : 0.8122
```

---

## Project Structure

```text
adaptive-serendipity-recommender/
│
├── main.py
├── requirements.txt
├── README.md
│
├── data/
│   └── feedback.csv
│
├── outputs/
│   ├── recommendations.csv
│   └── metrics.json
│
└── src/
    ├── Data_Loader.py
    ├── SVD_model.py
    ├── serendepity_engine.py
    ├── Evaluate.py
    └── rlhf.py
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/adaptive-serendipity-recommender.git
cd adaptive-serendipity-recommender
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

## RLHF-Inspired Adaptation

Unlike static recommendation systems, this project continuously adapts based on user feedback.

The system:

1. Generates recommendations.
2. Collects explicit user feedback.
3. Stores rewards in `feedback.csv`.
4. Learns a personalized alpha value.
5. Adjusts future recommendations accordingly.

This creates a simple but effective human-in-the-loop recommendation framework.

---

## Future Improvements

* Contextual Bandits
* Online Learning
* User Embeddings
* Implicit Feedback Signals
* Reinforcement Learning-based Policy Optimization
* Web Interface using Flask or Streamlit

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Surprise
* Matplotlib
* MovieLens Dataset

---

## Author

**Pranjali Manek**

B.Tech–M.Tech Integrated Program
The LNM Institute of Information Technology (LNMIIT), Jaipur

Interests:

* Machine Learning
* Recommendation Systems
* Reinforcement Learning
* NLP
* Game Development

---

## License

This project is intended for educational and research purposes.
