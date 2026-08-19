# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# # AMLAI - Aula 1 - Exercise 1: Build and Evaluate Neighbourhood Recommenders
#
# In Tutorial 1, the rating matrix used **users as rows** and **items as
# columns**. Keep that convention throughout this exercise.
#
# You will:
#
# 1. compute adjusted-cosine item similarities without treating missing
#    ratings as zeros;
# 2. generate normalized item-based predictions for unseen items;
# 3. build a positive-neighbour user-based recommender;
# 4. mask observed ratings and measure holdout RMSE and MAE.
#
# Replace each `____` placeholder, then run the checks below the task. Hidden
# reference solutions are available after each section, but try the task first.

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

# The raw source uses 0 only as an input marker for "not rated".
raw_ratings = np.array(
    [
        [5, 5, 0, 1, 4, 0, 2, 0],
        [4, 5, 0, 1, 5, 4, 0, 3],
        [0, 2, 5, 5, 0, 4, 4, 4],
        [1, 0, 4, 5, 2, 4, 5, 0],
        [5, 4, 2, 0, 5, 0, 2, 3],
        [0, 1, 5, 4, 2, 5, 5, 4],
    ],
    dtype=float,
)

# Convert missing markers immediately. All later calculations use NaN masks.
raw_ratings[raw_ratings == 0] = np.nan
ratings_df = pd.DataFrame(raw_ratings, index=users, columns=movies)

print("Exercise rating matrix (NaN = not rated):")
print(ratings_df)
print(f"\nObserved ratings: {int(ratings_df.notna().sum().sum())}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def validate_ratings(ratings):
    """Validate a user-row/item-column rating matrix."""
    if not isinstance(ratings, pd.DataFrame) or ratings.empty:
        raise ValueError("ratings must be a non-empty pandas DataFrame.")
    if not ratings.index.is_unique or not ratings.columns.is_unique:
        raise ValueError("User and item labels must be unique.")

    values = ratings.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Ratings must be finite values or NaN.")
    observed = ~np.isnan(values)
    if not observed.any():
        raise ValueError("The matrix has no observed ratings.")
    if (~observed.any(axis=1)).any():
        names = ratings.index[~observed.any(axis=1)].tolist()
        raise ValueError(f"Users without ratings are unsupported: {names}")
    if (~observed.any(axis=0)).any():
        names = ratings.columns[~observed.any(axis=0)].tolist()
        raise ValueError(f"Items without ratings are unsupported: {names}")

    observed_values = values[observed]
    if ((observed_values < RATING_MIN) | (observed_values > RATING_MAX)).any():
        raise ValueError("Observed ratings must be in the interval [1, 5].")


def make_user_holdout(ratings, seed=17):
    """
    Mask at most one observed rating per user without emptying a user or item.

    Returns a training matrix plus a table of held-out user-item-rating rows.
    """
    validate_ratings(ratings)
    rng = np.random.default_rng(seed)
    train = ratings.copy()
    item_counts = train.notna().sum(axis=0).astype(int)
    held_out = []
    skipped_users = []

    for user in rng.permutation(train.index.to_numpy()):
        observed_items = train.columns[train.loc[user].notna()].to_numpy()
        viable_items = [
            item for item in observed_items if item_counts.loc[item] > 1
        ]
        if len(observed_items) < 2 or not viable_items:
            skipped_users.append(user)
            continue

        item = str(rng.choice(np.array(viable_items, dtype=object)))
        rating = float(train.loc[user, item])
        train.loc[user, item] = np.nan
        item_counts.loc[item] -= 1
        held_out.append({"user": user, "item": item, "rating": rating})

    if not held_out:
        raise ValueError(
            "No safe holdout could be created; add denser user-item evidence."
        )
    if skipped_users:
        print(
            "No safe holdout for these sparse users: "
            + ", ".join(map(str, skipped_users))
        )

    validate_ratings(train)
    return train, pd.DataFrame(held_out)


def rating_errors(actual, predicted):
    """Return RMSE and MAE for finite paired predictions."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("actual and predicted must be non-empty equal shapes.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite actual and predicted values.")

    errors = actual - predicted
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    return rmse, mae


# The setup itself is deterministic and can be inspected before solving tasks.
example_train_df, example_holdout_df = make_user_holdout(ratings_df, seed=17)
print("\nExample training mask:")
print(example_train_df)
print("\nHeld-out observations:")
print(example_holdout_df.to_string(index=False))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 1. Adjusted-Cosine Item Similarity
#
# Complete the function below. Use only users who rated both items. Require at
# least `min_common` co-raters and guard against a zero denominator.
#
# Remember: because users are rows, an item's rating vector is a **column**.

# CELL ********************

def adjusted_cosine_item_similarity(ratings, min_common=2):
    validate_ratings(ratings)
    if min_common < 1:
        raise ValueError("min_common must be at least 1.")

    user_means = ____
    centred = ____
    item_names = ratings.columns
    similarities = np.zeros((len(item_names), len(item_names)), dtype=float)

    for i, left_item in enumerate(item_names):
        similarities[i, i] = 1.0
        for j in range(i + 1, len(item_names)):
            right_item = item_names[j]
            common = ____
            if int(common.sum()) < min_common:
                continue

            left = centred.loc[common, left_item].to_numpy()
            right = centred.loc[common, right_item].to_numpy()
            denominator = ____
            if denominator > 0:
                value = ____
                similarities[i, j] = similarities[j, i] = value

    return pd.DataFrame(
        similarities, index=item_names, columns=item_names
    )


item_similarity_df = adjusted_cosine_item_similarity(ratings_df)

assert item_similarity_df.shape == (len(movies), len(movies))
assert np.allclose(item_similarity_df, item_similarity_df.T)
assert np.allclose(np.diag(item_similarity_df), 1.0)
assert np.isfinite(item_similarity_df.to_numpy()).all()
print("Task 1 checks passed.")
print(item_similarity_df.round(3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# <details>
# <summary>Reference solution for Task 1</summary>
#
# ```python
# def adjusted_cosine_item_similarity(ratings, min_common=2):
#     validate_ratings(ratings)
#     if min_common < 1:
#         raise ValueError("min_common must be at least 1.")
#
#     user_means = ratings.mean(axis=1, skipna=True)
#     centred = ratings.sub(user_means, axis=0)
#     item_names = ratings.columns
#     similarities = np.zeros((len(item_names), len(item_names)), dtype=float)
#
#     for i, left_item in enumerate(item_names):
#         similarities[i, i] = 1.0
#         for j in range(i + 1, len(item_names)):
#             right_item = item_names[j]
#             common = ratings[left_item].notna() & ratings[right_item].notna()
#             if int(common.sum()) < min_common:
#                 continue
#             left = centred.loc[common, left_item].to_numpy()
#             right = centred.loc[common, right_item].to_numpy()
#             denominator = np.linalg.norm(left) * np.linalg.norm(right)
#             if denominator > 0:
#                 value = float(np.dot(left, right) / denominator)
#                 similarities[i, j] = similarities[j, i] = value
#
#     return pd.DataFrame(similarities, index=item_names, columns=item_names)
# ```
# </details>

# MARKDOWN ********************

# ## 2. Item-Based Recommendations
#
# Predict an unseen rating with a normalized weighted sum of the target user's
# rating deviations. Keep only positive similarities. Then create the unseen
# candidate set **before** sorting so that already-rated movies cannot consume
# a top-\(k\) position.

# CELL ********************

def predict_item_rating(user, item, ratings, item_similarities):
    if user not in ratings.index:
        raise KeyError(f"Unknown user: {user!r}")
    if item not in ratings.columns:
        raise KeyError(f"Unknown item: {item!r}")
    if not pd.isna(ratings.loc[user, item]):
        raise ValueError(f"{user!r} has already rated {item!r}.")

    user_ratings = ratings.loc[user]
    rated_items = ____
    similarities = ____
    useful = ____
    if useful.empty:
        return None

    user_mean = float(user_ratings.mean())
    deviations = user_ratings.loc[useful.index] - user_mean
    denominator = ____
    if denominator <= 0:
        return None

    prediction = ____
    return float(np.clip(prediction, RATING_MIN, RATING_MAX))


def recommend_items(user, ratings, item_similarities, k=3):
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    if user not in ratings.index:
        raise KeyError(f"Unknown user: {user!r}")

    unseen_items = ____
    if len(unseen_items) == 0:
        raise ValueError(f"{user!r} has no unseen items to recommend.")

    valid_predictions = []
    for item in unseen_items:
        prediction = ____
        if prediction is not None:
            valid_predictions.append((item, prediction))

    if not valid_predictions:
        raise ValueError(
            f"No unseen item for {user!r} has a useful item neighbour."
        )

    valid_predictions.sort(key=lambda pair: (-pair[1], pair[0]))
    return pd.DataFrame(
        valid_predictions[:k], columns=["item", "predicted_rating"]
    )


item_recommendations = recommend_items(
    "Bob", ratings_df, item_similarity_df, k=3
)
already_rated = ratings_df.columns[ratings_df.loc["Bob"].notna()]

assert 1 <= len(item_recommendations) <= 3
assert not item_recommendations["item"].isin(already_rated).any()
assert item_recommendations["predicted_rating"].between(1, 5).all()
print("Task 2 checks passed.")
print(item_recommendations.to_string(index=False))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# <details>
# <summary>Reference solution for Task 2</summary>
#
# ```python
# def predict_item_rating(user, item, ratings, item_similarities):
#     if user not in ratings.index:
#         raise KeyError(f"Unknown user: {user!r}")
#     if item not in ratings.columns:
#         raise KeyError(f"Unknown item: {item!r}")
#     if not pd.isna(ratings.loc[user, item]):
#         raise ValueError(f"{user!r} has already rated {item!r}.")
#
#     user_ratings = ratings.loc[user]
#     rated_items = user_ratings.dropna().index
#     similarities = item_similarities.loc[item, rated_items]
#     useful = similarities[similarities > 0]
#     if useful.empty:
#         return None
#     user_mean = float(user_ratings.mean())
#     deviations = user_ratings.loc[useful.index] - user_mean
#     denominator = float(useful.abs().sum())
#     if denominator <= 0:
#         return None
#     prediction = user_mean + float(np.dot(useful, deviations)) / denominator
#     return float(np.clip(prediction, RATING_MIN, RATING_MAX))
#
# def recommend_items(user, ratings, item_similarities, k=3):
#     if not isinstance(k, int) or k < 1:
#         raise ValueError("k must be a positive integer.")
#     if user not in ratings.index:
#         raise KeyError(f"Unknown user: {user!r}")
#     unseen_items = ratings.columns[ratings.loc[user].isna()]
#     if len(unseen_items) == 0:
#         raise ValueError(f"{user!r} has no unseen items to recommend.")
#     valid_predictions = []
#     for item in unseen_items:
#         prediction = predict_item_rating(
#             user, item, ratings, item_similarities
#         )
#         if prediction is not None:
#             valid_predictions.append((item, prediction))
#     if not valid_predictions:
#         raise ValueError(
#             f"No unseen item for {user!r} has a useful item neighbour."
#         )
#     valid_predictions.sort(key=lambda pair: (-pair[1], pair[0]))
#     return pd.DataFrame(
#         valid_predictions[:k], columns=["item", "predicted_rating"]
#     )
# ```
# </details>

# MARKDOWN ********************

# ## 3. User-Based Recommendations
#
# Calculate Pearson similarity using co-rated items only. The recommendation
# function must exclude the active user and select up to `k_neighbors`
# positive neighbours that rated each candidate item.

# CELL ********************

def pearson_user_similarity(ratings, min_common=2):
    validate_ratings(ratings)
    if min_common < 2:
        raise ValueError("Pearson correlation requires min_common >= 2.")
    user_names = ratings.index
    similarities = np.zeros((len(user_names), len(user_names)), dtype=float)

    for i, left_user in enumerate(user_names):
        similarities[i, i] = 1.0
        for j in range(i + 1, len(user_names)):
            right_user = user_names[j]
            common = ____
            if int(common.sum()) < min_common:
                continue

            left = ratings.loc[left_user, common].to_numpy(dtype=float)
            right = ratings.loc[right_user, common].to_numpy(dtype=float)
            if np.std(left) == 0 or np.std(right) == 0:
                continue

            value = ____
            if np.isfinite(value):
                similarities[i, j] = similarities[j, i] = value

    return pd.DataFrame(
        similarities, index=user_names, columns=user_names
    )


def recommend_users(user, ratings, user_similarities, k=3, k_neighbors=3):
    validate_ratings(ratings)
    if user not in ratings.index:
        raise KeyError(f"Unknown user: {user!r}")
    if (
        not isinstance(k, int)
        or not isinstance(k_neighbors, int)
        or k < 1
        or k_neighbors < 1
    ):
        raise ValueError("k and k_neighbors must be positive integers.")

    neighbours = ____
    neighbours = ____
    if neighbours.empty:
        raise ValueError(f"{user!r} has no positive user neighbours.")

    unseen_items = ratings.columns[ratings.loc[user].isna()]
    if len(unseen_items) == 0:
        raise ValueError(f"{user!r} has no unseen items to recommend.")

    target_mean = float(ratings.loc[user].mean())
    predictions = []
    for item in unseen_items:
        contributors = ____
        if len(contributors) == 0:
            continue

        weights = neighbours.loc[contributors]
        neighbour_means = ratings.loc[contributors].mean(axis=1)
        deviations = ratings.loc[contributors, item] - neighbour_means
        denominator = float(weights.sum())
        if denominator > 0:
            prediction = ____
            predictions.append(
                (item, float(np.clip(prediction, RATING_MIN, RATING_MAX)))
            )

    if not predictions:
        raise ValueError(
            f"No unseen item for {user!r} was rated by a positive neighbour."
        )
    predictions.sort(key=lambda pair: (-pair[1], pair[0]))
    return pd.DataFrame(
        predictions[:k], columns=["item", "predicted_rating"]
    )


user_similarity_df = pearson_user_similarity(ratings_df)
user_recommendations = recommend_users(
    "Eve", ratings_df, user_similarity_df, k=3, k_neighbors=3
)

assert 1 <= len(user_recommendations) <= 3
assert not user_recommendations["item"].isin(
    ratings_df.columns[ratings_df.loc["Eve"].notna()]
).any()
assert user_recommendations["predicted_rating"].between(1, 5).all()
print("Task 3 checks passed.")
print(user_recommendations.to_string(index=False))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# <details>
# <summary>Reference solution for Task 3</summary>
#
# ```python
# def pearson_user_similarity(ratings, min_common=2):
#     validate_ratings(ratings)
#     if min_common < 2:
#         raise ValueError("Pearson correlation requires min_common >= 2.")
#     user_names = ratings.index
#     similarities = np.zeros((len(user_names), len(user_names)), dtype=float)
#     for i, left_user in enumerate(user_names):
#         similarities[i, i] = 1.0
#         for j in range(i + 1, len(user_names)):
#             right_user = user_names[j]
#             common = (
#                 ratings.loc[left_user].notna()
#                 & ratings.loc[right_user].notna()
#             )
#             if int(common.sum()) < min_common:
#                 continue
#             left = ratings.loc[left_user, common].to_numpy(dtype=float)
#             right = ratings.loc[right_user, common].to_numpy(dtype=float)
#             if np.std(left) == 0 or np.std(right) == 0:
#                 continue
#             value = float(np.corrcoef(left, right)[0, 1])
#             if np.isfinite(value):
#                 similarities[i, j] = similarities[j, i] = value
#     return pd.DataFrame(
#         similarities, index=user_names, columns=user_names
#     )
#
# def recommend_users(user, ratings, user_similarities, k=3, k_neighbors=3):
#     validate_ratings(ratings)
#     if user not in ratings.index:
#         raise KeyError(f"Unknown user: {user!r}")
#     if (
#         not isinstance(k, int)
#         or not isinstance(k_neighbors, int)
#         or k < 1
#         or k_neighbors < 1
#     ):
#         raise ValueError("k and k_neighbors must be positive integers.")
#     neighbours = user_similarities.loc[user].drop(index=user)
#     neighbours = neighbours[neighbours > 0]
#     if neighbours.empty:
#         raise ValueError(f"{user!r} has no positive user neighbours.")
#     unseen_items = ratings.columns[ratings.loc[user].isna()]
#     if len(unseen_items) == 0:
#         raise ValueError(f"{user!r} has no unseen items to recommend.")
#     target_mean = float(ratings.loc[user].mean())
#     predictions = []
#     for item in unseen_items:
#         contributors = neighbours[
#             ratings.loc[neighbours.index, item].notna()
#         ].nlargest(k_neighbors).index
#         if len(contributors) == 0:
#             continue
#         weights = neighbours.loc[contributors]
#         neighbour_means = ratings.loc[contributors].mean(axis=1)
#         deviations = ratings.loc[contributors, item] - neighbour_means
#         denominator = float(weights.sum())
#         if denominator > 0:
#             prediction = (
#                 target_mean + float(np.dot(weights, deviations)) / denominator
#             )
#             predictions.append(
#                 (item, float(np.clip(prediction, RATING_MIN, RATING_MAX)))
#             )
#     if not predictions:
#         raise ValueError(
#             f"No unseen item for {user!r} was rated by a positive neighbour."
#         )
#     predictions.sort(key=lambda pair: (-pair[1], pair[0]))
#     return pd.DataFrame(
#         predictions[:k], columns=["item", "predicted_rating"]
#     )
# ```
# </details>

# MARKDOWN ********************

# ## 4. Holdout Evaluation
#
# Offline evaluation must hide ratings before fitting similarities. Otherwise,
# the answer leaks into the model. Use the deterministic masking helper from
# the setup, fit item similarity on `train_ratings`, predict each held-out
# observation, and calculate RMSE and MAE only for valid predictions.
#
# Some sparse holdouts may have no positive item neighbour. Record those as
# unavailable rather than inventing a rating, and report prediction coverage.

# CELL ********************

train_ratings, held_out = make_user_holdout(ratings_df, seed=17)
train_item_similarity = ____

evaluation_rows = []
for row in held_out.itertuples(index=False):
    predicted = ____
    evaluation_rows.append(
        {
            "user": row.user,
            "item": row.item,
            "actual": row.rating,
            "predicted": predicted,
        }
    )

evaluation_df = pd.DataFrame(evaluation_rows)
scored_df = evaluation_df.dropna(subset=["predicted"])
if scored_df.empty:
    raise ValueError(
        "No held-out rating had enough positive-neighbour evidence to score."
    )

rmse, mae = ____
coverage = len(scored_df) / len(evaluation_df)

assert np.isfinite(rmse) and np.isfinite(mae)
assert 0 < coverage <= 1
print("Task 4 checks passed.")
print(evaluation_df.to_string(index=False))
print(f"\nHoldout RMSE: {rmse:.3f}")
print(f"Holdout MAE:  {mae:.3f}")
print(f"Coverage:     {coverage:.1%}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# <details>
# <summary>Reference solution for Task 4</summary>
#
# ```python
# train_ratings, held_out = make_user_holdout(ratings_df, seed=17)
# train_item_similarity = adjusted_cosine_item_similarity(train_ratings)
#
# evaluation_rows = []
# for row in held_out.itertuples(index=False):
#     predicted = predict_item_rating(
#         row.user, row.item, train_ratings, train_item_similarity
#     )
#     evaluation_rows.append(
#         {
#             "user": row.user,
#             "item": row.item,
#             "actual": row.rating,
#             "predicted": predicted,
#         }
#     )
#
# evaluation_df = pd.DataFrame(evaluation_rows)
# scored_df = evaluation_df.dropna(subset=["predicted"])
# if scored_df.empty:
#     raise ValueError(
#         "No held-out rating had enough positive-neighbour evidence to score."
#     )
# rmse, mae = rating_errors(scored_df["actual"], scored_df["predicted"])
# coverage = len(scored_df) / len(evaluation_df)
# ```
# </details>

# MARKDOWN ********************

# ## Reflection
#
# - Why is coverage useful alongside RMSE and MAE?
# - What changes when `min_common` or `k_neighbors` increases?
# - Why would evaluating on ratings used to calculate similarity be optimistic?
# - What side information could help a genuinely new user or new movie?
#
# Tutorial 2 introduces regularized matrix factorization. It improves how
# sparse observed ratings are modelled, but it still needs side information or
# an onboarding strategy for true cold start.
