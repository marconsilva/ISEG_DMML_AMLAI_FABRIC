# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# # AMLAI - Aula 1 - Tutorial 1: Neighbourhood-Based Recommendation Systems
#
# Recommendation systems rank items that may be useful to a user. In this
# tutorial we use explicit movie ratings and two collaborative-filtering
# approaches:
#
# 1. **Item-based filtering**: recommend items that resemble items the user
#    has rated.
# 2. **User-based filtering**: recommend items liked by users with similar
#    rating patterns.
#
# Throughout Aula 1 we use one convention consistently:
#
# - **rows are users**;
# - **columns are items**;
# - a numeric value is an observed rating;
# - `NaN` means "not rated".
#
# A missing rating is unknown, not a zero-star opinion. It must therefore be
# excluded from means, similarities, training losses, and evaluation metrics.

# MARKDOWN ********************

# ## 1. Similarity with Missing Ratings
#
# Similarity must be calculated only where evidence exists.
#
# For item-based filtering we use **adjusted cosine similarity**. First, each
# observed rating is centred by that user's mean rating. For items \(i\) and
# \(j\), let \(U_{ij}\) be the users who rated both:
#
# $$
# s(i,j)=
# \frac{\sum_{u \in U_{ij}}(r_{ui}-\bar r_u)(r_{uj}-\bar r_u)}
# {\sqrt{\sum_{u \in U_{ij}}(r_{ui}-\bar r_u)^2}
#  \sqrt{\sum_{u \in U_{ij}}(r_{uj}-\bar r_u)^2}}.
# $$
#
# This avoids treating missing values as ratings and reduces the effect of
# users who systematically rate high or low. A pair with too little overlap
# or zero variance receives similarity `0.0`: there is not enough useful
# evidence to use that pair as a neighbour.
#
# For user-based filtering we use **Pearson correlation** on co-rated items
# only. We later keep positive neighbours: a negative correlation describes
# opposing preferences and is deliberately not used by this introductory
# predictor.

# CELL ********************

import numpy as np
import pandas as pd

RATING_MIN = 1.0
RATING_MAX = 5.0

users = ["Alice", "Bob", "Carol", "David", "Eve", "Frank"]
movies = [
    "Inception",
    "The Matrix",
    "Titanic",
    "The Notebook",
    "Interstellar",
    "Pulp Fiction",
    "Forrest Gump",
    "The Godfather",
]

# Rows are users, columns are movies, and np.nan means "not rated".
ratings_df = pd.DataFrame(
    [
        [5, 5, np.nan, 1, 4, np.nan, 2, np.nan],
        [4, 5, np.nan, 1, 5, 4, np.nan, 3],
        [np.nan, 2, 5, 5, np.nan, 4, 4, 4],
        [1, np.nan, 4, 5, 2, 4, 5, np.nan],
        [5, 4, 2, np.nan, 5, np.nan, 2, 3],
        [np.nan, 1, 5, 4, 2, 5, 5, 4],
    ],
    index=users,
    columns=movies,
    dtype=float,
)

print("User-item rating matrix (NaN = not rated):")
print(ratings_df)
observed = int(ratings_df.notna().sum().sum())
print(f"\nObserved ratings: {observed}/{ratings_df.size}")
print(f"Sparsity: {ratings_df.isna().to_numpy().mean():.1%}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def validate_ratings(ratings):
    """Validate a user-row/item-column rating DataFrame."""
    if not isinstance(ratings, pd.DataFrame) or ratings.empty:
        raise ValueError("ratings must be a non-empty pandas DataFrame.")
    if not ratings.index.is_unique or not ratings.columns.is_unique:
        raise ValueError("User and item labels must be unique.")

    values = ratings.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Ratings must be finite values or NaN.")
    observed_mask = ~np.isnan(values)
    if not observed_mask.any():
        raise ValueError("The matrix has no observed ratings.")
    if (~observed_mask.any(axis=1)).any():
        empty_users = ratings.index[~observed_mask.any(axis=1)].tolist()
        raise ValueError(f"Users without ratings are unsupported: {empty_users}")
    if (~observed_mask.any(axis=0)).any():
        empty_items = ratings.columns[~observed_mask.any(axis=0)].tolist()
        raise ValueError(f"Items without ratings are unsupported: {empty_items}")

    observed_values = values[observed_mask]
    if ((observed_values < RATING_MIN) | (observed_values > RATING_MAX)).any():
        raise ValueError(
            f"Observed ratings must be between {RATING_MIN} and {RATING_MAX}."
        )


def adjusted_cosine_item_similarity(ratings, min_common=2):
    """Return item-item adjusted-cosine similarities and overlap counts."""
    validate_ratings(ratings)
    if min_common < 1:
        raise ValueError("min_common must be at least 1.")

    user_means = ratings.mean(axis=1, skipna=True)
    centred = ratings.sub(user_means, axis=0)
    item_names = ratings.columns
    n_items = len(item_names)
    similarities = np.zeros((n_items, n_items), dtype=float)
    overlaps = np.zeros((n_items, n_items), dtype=int)

    for i in range(n_items):
        for j in range(i, n_items):
            common = ratings.iloc[:, i].notna() & ratings.iloc[:, j].notna()
            overlap = int(common.sum())
            overlaps[i, j] = overlaps[j, i] = overlap

            if i == j:
                similarities[i, j] = 1.0
                continue
            if overlap < min_common:
                continue

            left = centred.loc[common, item_names[i]].to_numpy()
            right = centred.loc[common, item_names[j]].to_numpy()
            denominator = np.linalg.norm(left) * np.linalg.norm(right)
            if denominator > 0:
                value = float(np.dot(left, right) / denominator)
                similarities[i, j] = similarities[j, i] = value

    return (
        pd.DataFrame(similarities, index=item_names, columns=item_names),
        pd.DataFrame(overlaps, index=item_names, columns=item_names),
    )


item_similarity_df, item_overlap_df = adjusted_cosine_item_similarity(ratings_df)

print("Adjusted-cosine item similarities:")
print(item_similarity_df.round(3))
print("\nNumber of co-rating users for each item pair:")
print(item_overlap_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 2. Item-Based Rating Prediction
#
# For an unseen item \(i\), use items \(j\) already rated by user \(u\):
#
# $$
# \hat r_{ui}=\bar r_u+
# \frac{\sum_{j \in N_u(i)}s(i,j)(r_{uj}-\bar r_u)}
# {\sum_{j \in N_u(i)}|s(i,j)|}.
# $$
#
# The rating deviation is weighted by similarity and the result is normalized.
# This is different from merely adding similarity values. Here we retain only
# positive similarities so that each neighbour supplies useful evidence.
# A candidate with a zero denominator is not silently assigned a fabricated
# score; it is omitted.
#
# Candidate items are filtered to unseen items **before** ranking. The returned
# table therefore contains exactly `min(k, number of valid unseen candidates)`
# rows.

# CELL ********************

def predict_item_based(user, item, ratings, item_similarities):
    """Predict one unseen rating; return None when no useful evidence exists."""
    validate_ratings(ratings)
    if user not in ratings.index:
        raise KeyError(f"Unknown user: {user!r}")
    if item not in ratings.columns:
        raise KeyError(f"Unknown item: {item!r}")
    if not pd.isna(ratings.loc[user, item]):
        raise ValueError(f"{user!r} has already rated {item!r}.")

    user_ratings = ratings.loc[user]
    rated_items = user_ratings.dropna().index
    similarities = item_similarities.loc[item, rated_items]
    useful = similarities[similarities > 0]
    if useful.empty:
        return None

    user_mean = float(user_ratings.mean())
    deviations = user_ratings.loc[useful.index] - user_mean
    denominator = float(useful.abs().sum())
    if denominator <= 0:
        return None

    prediction = user_mean + float(np.dot(useful, deviations)) / denominator
    return float(np.clip(prediction, RATING_MIN, RATING_MAX))


def recommend_item_based(user, ratings, item_similarities, k=3):
    """Rank up to k valid unseen items with item-based predictions."""
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    if user not in ratings.index:
        raise KeyError(f"Unknown user: {user!r}")

    unseen_items = ratings.columns[ratings.loc[user].isna()]
    if len(unseen_items) == 0:
        raise ValueError(f"{user!r} has no unseen items to recommend.")

    scored = []
    for item in unseen_items:
        prediction = predict_item_based(
            user, item, ratings, item_similarities
        )
        if prediction is not None:
            scored.append((item, prediction))

    if not scored:
        raise ValueError(
            f"No unseen item for {user!r} has a positive-similarity neighbour."
        )

    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    selected = scored[:k]
    if len(selected) < k:
        print(
            f"Only {len(selected)} valid unseen item(s) are available "
            f"for {user!r}; requested {k}."
        )
    return pd.DataFrame(selected, columns=["item", "predicted_rating"])


item_recommendations = recommend_item_based(
    "Bob", ratings_df, item_similarity_df, k=3
)
print("Item-based recommendations for Bob:")
print(item_recommendations.to_string(index=False))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 3. User-Based Collaborative Filtering
#
# User similarity is calculated on items rated by both users. For target user
# \(u\), the predictor excludes \(u\) from the neighbour set, keeps up to
# `k_neighbors` users with positive Pearson correlation, and predicts:
#
# $$
# \hat r_{ui}=\bar r_u+
# \frac{\sum_{v \in N_u(i)}s(u,v)(r_{vi}-\bar r_v)}
# {\sum_{v \in N_u(i)}s(u,v)}.
# $$
#
# A neighbour contributes only if they rated candidate item \(i\). Negative and
# zero similarities are deliberately excluded in this introductory version.

# CELL ********************

def pearson_user_similarity(ratings, min_common=2):
    """Return user-user Pearson similarities using only co-rated items."""
    validate_ratings(ratings)
    if min_common < 2:
        raise ValueError("Pearson correlation requires min_common >= 2.")

    user_names = ratings.index
    n_users = len(user_names)
    similarities = np.zeros((n_users, n_users), dtype=float)
    overlaps = np.zeros((n_users, n_users), dtype=int)

    for i in range(n_users):
        similarities[i, i] = 1.0
        overlaps[i, i] = int(ratings.iloc[i].notna().sum())
        for j in range(i + 1, n_users):
            common = ratings.iloc[i].notna() & ratings.iloc[j].notna()
            overlap = int(common.sum())
            overlaps[i, j] = overlaps[j, i] = overlap
            if overlap < min_common:
                continue

            left = ratings.iloc[i][common].to_numpy(dtype=float)
            right = ratings.iloc[j][common].to_numpy(dtype=float)
            if np.std(left) == 0 or np.std(right) == 0:
                continue

            value = float(np.corrcoef(left, right)[0, 1])
            if np.isfinite(value):
                similarities[i, j] = similarities[j, i] = value

    return (
        pd.DataFrame(similarities, index=user_names, columns=user_names),
        pd.DataFrame(overlaps, index=user_names, columns=user_names),
    )


def predict_user_based(
    user, item, ratings, user_similarities, k_neighbors=3
):
    """Predict one unseen rating from positive user neighbours."""
    validate_ratings(ratings)
    if user not in ratings.index:
        raise KeyError(f"Unknown user: {user!r}")
    if item not in ratings.columns:
        raise KeyError(f"Unknown item: {item!r}")
    if not pd.isna(ratings.loc[user, item]):
        raise ValueError(f"{user!r} has already rated {item!r}.")
    if not isinstance(k_neighbors, int) or k_neighbors < 1:
        raise ValueError("k_neighbors must be a positive integer.")

    neighbours = user_similarities.loc[user].drop(index=user)
    neighbours = neighbours[
        (neighbours > 0) & ratings.loc[neighbours.index, item].notna()
    ].nlargest(k_neighbors)
    if neighbours.empty:
        return None

    contributors = neighbours.index
    weights = neighbours
    neighbour_means = ratings.loc[contributors].mean(axis=1)
    deviations = ratings.loc[contributors, item] - neighbour_means
    denominator = float(weights.sum())
    if denominator <= 0:
        return None

    target_mean = float(ratings.loc[user].mean())
    prediction = target_mean + float(np.dot(weights, deviations)) / denominator
    return float(np.clip(prediction, RATING_MIN, RATING_MAX))


def recommend_user_based(
    user, ratings, user_similarities, k=3, k_neighbors=3
):
    """Rank up to k valid unseen items using positive user neighbours."""
    validate_ratings(ratings)
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    if user not in ratings.index:
        raise KeyError(f"Unknown user: {user!r}")

    positive_neighbours = user_similarities.loc[user].drop(index=user)
    positive_neighbours = positive_neighbours[positive_neighbours > 0]
    if positive_neighbours.empty:
        raise ValueError(f"{user!r} has no positive user neighbours.")

    unseen_items = ratings.columns[ratings.loc[user].isna()]
    if len(unseen_items) == 0:
        raise ValueError(f"{user!r} has no unseen items to recommend.")

    scored = []
    for item in unseen_items:
        prediction = predict_user_based(
            user,
            item,
            ratings,
            user_similarities,
            k_neighbors=k_neighbors,
        )
        if prediction is not None:
            scored.append((item, prediction))

    if not scored:
        raise ValueError(
            f"No unseen item for {user!r} was rated by a positive neighbour."
        )

    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    selected = scored[:k]
    if len(selected) < k:
        print(
            f"Only {len(selected)} valid unseen item(s) are available "
            f"for {user!r}; requested {k}."
        )
    return pd.DataFrame(selected, columns=["item", "predicted_rating"])


user_similarity_df, user_overlap_df = pearson_user_similarity(ratings_df)
print("Pearson user similarities:")
print(user_similarity_df.round(3))

user_recommendations = recommend_user_based(
    "Eve", ratings_df, user_similarity_df, k=3, k_neighbors=3
)
print("\nUser-based recommendations for Eve:")
print(user_recommendations.to_string(index=False))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 4. What the Example Does — and Does Not Do
#
# - Similarities and means ignore missing ratings.
# - Item-based scores use ratings, similarity weights, and normalization.
# - User-based scores select positive neighbours that rated each candidate.
# - Already-rated items are removed before candidates are ranked.
# - Unsupported candidates are reported or omitted rather than given a
#   misleading score.
# - Collaborative filtering still has a **cold-start limitation**. A new user
#   or new item has no interaction history. Matrix factorization can model
#   sparse observed interactions efficiently, but it does not solve cold start
#   unless side information or an explicit onboarding strategy is added.
#
# Continue with **Exercise 1** to implement and evaluate these ideas. Tutorial 2
# then advances from neighbourhood methods to regularized matrix factorization.
