# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# # AMLAI - Aula 1 - Tutorial 2: Regularized Matrix Factorization
#
# Tutorial 1 represented the rating matrix with **users as rows** and **items
# as columns**. Neighbourhood methods compare those rows or columns directly.
# Matrix factorization instead learns a short latent vector for every known
# user and item.
#
# For \(m\) users, \(n\) items, and \(f\) latent factors:
#
# - \(P \in \mathbb{R}^{m \times f}\) contains user factors;
# - \(Q \in \mathbb{R}^{n \times f}\) contains item factors;
# - only observed ratings are used for training.
#
# The biased prediction is
#
# $$
# \hat r_{ui} = \mu + b_u + b_i + p_u^\top q_i,
# $$
#
# where \(\mu\) is the training-set mean and \(b_u\), \(b_i\) are user and item
# biases.
#
# Matrix factorization can model sparse observed data compactly and avoids
# storing a full user-user or item-item similarity matrix. It **does not solve
# true cold start by itself**: a user or item with no interactions has no
# learned factors. Side information, popularity fallbacks, or an onboarding
# process is still required.

# MARKDOWN ********************

# ## 1. Objective and Regularization
#
# We minimize squared error on the set \(\Omega_{\text{train}}\) of observed
# training ratings, plus an \(L_2\) penalty:
#
# $$
# \mathcal{L} =
# \sum_{(u,i)\in\Omega_{\text{train}}}
# \left(r_{ui}-\mu-b_u-b_i-p_u^\top q_i\right)^2
# + \lambda\left(
# b_u^2+b_i^2+\lVert p_u\rVert_2^2+\lVert q_i\rVert_2^2
# \right).
# $$
#
# Regularization discourages extreme parameters that merely memorize a small
# dataset. The example below uses deterministic stochastic gradient descent
# (SGD), a fixed random seed, and a validation split containing only ratings
# hidden from training.

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

print("User-item matrix (NaN = not rated):")
print(ratings_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 2. Reproducible Train/Validation Split
#
# The splitter considers observed ratings only. It hides ratings one by one
# while ensuring that every user and every item retains at least one training
# observation. On extremely sparse data, fewer validation rows than requested
# may be possible; the function reports that case explicitly.

# CELL ********************

def validate_ratings(ratings):
    """Validate a user-row/item-column matrix with NaN for missing ratings."""
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


def observed_train_validation_split(
    ratings, validation_fraction=0.2, seed=23
):
    """Mask a deterministic subset while retaining train support per row/column."""
    validate_ratings(ratings)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    observed_positions = np.argwhere(ratings.notna().to_numpy())
    if len(observed_positions) < 3:
        raise ValueError("At least three observed ratings are required.")

    target = max(1, int(round(len(observed_positions) * validation_fraction)))
    target = min(target, len(observed_positions) - 1)
    user_counts = ratings.notna().sum(axis=1).to_numpy(dtype=int).copy()
    item_counts = ratings.notna().sum(axis=0).to_numpy(dtype=int).copy()
    rng = np.random.default_rng(seed)

    train = ratings.copy()
    validation_rows = []
    for position_index in rng.permutation(len(observed_positions)):
        if len(validation_rows) == target:
            break

        user_pos, item_pos = observed_positions[position_index]
        if user_counts[user_pos] <= 1 or item_counts[item_pos] <= 1:
            continue

        user = ratings.index[user_pos]
        item = ratings.columns[item_pos]
        rating = float(ratings.iat[user_pos, item_pos])
        train.iat[user_pos, item_pos] = np.nan
        user_counts[user_pos] -= 1
        item_counts[item_pos] -= 1
        validation_rows.append(
            {"user": user, "item": item, "rating": rating}
        )

    if not validation_rows:
        raise ValueError(
            "No safe validation split exists; provide denser interactions."
        )
    if len(validation_rows) < target:
        print(
            f"Requested {target} validation ratings, but only "
            f"{len(validation_rows)} could be masked safely."
        )

    validate_ratings(train)
    return train, pd.DataFrame(validation_rows)


train_df, validation_df = observed_train_validation_split(
    ratings_df, validation_fraction=0.2, seed=23
)
print(f"Training ratings:   {int(train_df.notna().sum().sum())}")
print(f"Validation ratings: {len(validation_df)}")
print(validation_df.to_string(index=False))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 3. NumPy Implementation with Regularized SGD
#
# Each epoch visits every observed training rating in a seeded random order.
# The user bias, item bias, user factors, and item factors are updated in the
# direction that reduces regularized prediction error.
#
# The implementation validates sparse and degenerate inputs instead of
# silently fitting users or items with no training evidence.

# CELL ********************

class BiasedMatrixFactorization:
    """Small explicit-rating matrix factorization model using NumPy SGD."""

    def __init__(
        self,
        n_factors=3,
        learning_rate=0.02,
        regularization=0.05,
        epochs=600,
        seed=31,
    ):
        if not isinstance(n_factors, int) or n_factors < 1:
            raise ValueError("n_factors must be a positive integer.")
        if learning_rate <= 0 or regularization < 0:
            raise ValueError(
                "learning_rate must be positive and regularization non-negative."
            )
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer.")

        self.n_factors = n_factors
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.epochs = epochs
        self.seed = seed
        self.is_fitted = False

    def _raw_prediction(self, user_pos, item_pos):
        return (
            self.global_mean
            + self.user_bias[user_pos]
            + self.item_bias[item_pos]
            + np.dot(
                self.user_factors[user_pos],
                self.item_factors[item_pos],
            )
        )

    def fit(self, ratings, validation=None):
        validate_ratings(ratings)
        values = ratings.to_numpy(dtype=float)
        observed_positions = np.argwhere(np.isfinite(values))
        if len(observed_positions) < 2:
            raise ValueError("At least two observed training ratings are needed.")

        self.user_names = ratings.index.copy()
        self.item_names = ratings.columns.copy()
        self.user_to_pos = {
            name: position for position, name in enumerate(self.user_names)
        }
        self.item_to_pos = {
            name: position for position, name in enumerate(self.item_names)
        }
        self.global_mean = float(np.nanmean(values))

        rng = np.random.default_rng(self.seed)
        self.user_bias = np.zeros(len(self.user_names), dtype=float)
        self.item_bias = np.zeros(len(self.item_names), dtype=float)
        self.user_factors = rng.normal(
            0.0, 0.1, size=(len(self.user_names), self.n_factors)
        )
        self.item_factors = rng.normal(
            0.0, 0.1, size=(len(self.item_names), self.n_factors)
        )
        self.history = []

        for epoch in range(1, self.epochs + 1):
            for sample_index in rng.permutation(len(observed_positions)):
                user_pos, item_pos = observed_positions[sample_index]
                rating = values[user_pos, item_pos]
                prediction = self._raw_prediction(user_pos, item_pos)
                error = rating - prediction

                old_user_factors = self.user_factors[user_pos].copy()
                old_item_factors = self.item_factors[item_pos].copy()
                self.user_bias[user_pos] += self.learning_rate * (
                    error
                    - self.regularization * self.user_bias[user_pos]
                )
                self.item_bias[item_pos] += self.learning_rate * (
                    error
                    - self.regularization * self.item_bias[item_pos]
                )
                self.user_factors[user_pos] += self.learning_rate * (
                    error * old_item_factors
                    - self.regularization * old_user_factors
                )
                self.item_factors[item_pos] += self.learning_rate * (
                    error * old_user_factors
                    - self.regularization * old_item_factors
                )

            train_predictions = np.array(
                [
                    self._raw_prediction(user_pos, item_pos)
                    for user_pos, item_pos in observed_positions
                ]
            )
            train_actual = values[
                observed_positions[:, 0], observed_positions[:, 1]
            ]
            train_rmse = float(
                np.sqrt(np.mean((train_actual - train_predictions) ** 2))
            )
            record = {"epoch": epoch, "train_rmse": train_rmse}

            if validation is not None:
                validation_predictions = [
                    self.predict(row.user, row.item, clip=True)
                    for row in validation.itertuples(index=False)
                ]
                record["validation_rmse"] = rating_rmse(
                    validation["rating"], validation_predictions
                )
            self.history.append(record)

        self.is_fitted = True
        return self

    def predict(self, user, item, clip=True):
        if not hasattr(self, "user_to_pos"):
            raise RuntimeError("Fit the model before requesting predictions.")
        if user not in self.user_to_pos:
            raise KeyError(
                f"Unknown user {user!r}: matrix factorization has no "
                "cold-start user factors."
            )
        if item not in self.item_to_pos:
            raise KeyError(
                f"Unknown item {item!r}: matrix factorization has no "
                "cold-start item factors."
            )

        value = float(
            self._raw_prediction(
                self.user_to_pos[user], self.item_to_pos[item]
            )
        )
        if clip:
            value = float(np.clip(value, RATING_MIN, RATING_MAX))
        return value


def rating_rmse(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("actual and predicted must be non-empty equal shapes.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("RMSE requires finite actual and predicted values.")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def rating_mae(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("actual and predicted must be non-empty equal shapes.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("MAE requires finite actual and predicted values.")
    return float(np.mean(np.abs(actual - predicted)))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 4. Train and Evaluate on Held-Out Ratings
#
# The validation ratings remain `NaN` in `train_df`; they never contribute to
# SGD updates or the training mean. RMSE penalizes larger errors more strongly,
# while MAE reports the average absolute miss in rating units.

# CELL ********************

model = BiasedMatrixFactorization(
    n_factors=3,
    learning_rate=0.02,
    regularization=0.05,
    epochs=600,
    seed=31,
)
model.fit(train_df, validation=validation_df)

validation_predictions = np.array(
    [
        model.predict(row.user, row.item)
        for row in validation_df.itertuples(index=False)
    ]
)
validation_results = validation_df.copy()
validation_results["predicted"] = validation_predictions
validation_results["absolute_error"] = np.abs(
    validation_results["rating"] - validation_results["predicted"]
)

validation_rmse = rating_rmse(
    validation_results["rating"], validation_results["predicted"]
)
validation_mae = rating_mae(
    validation_results["rating"], validation_results["predicted"]
)

assert np.isfinite(validation_rmse) and np.isfinite(validation_mae)
print(validation_results.round(3).to_string(index=False))
print(f"\nValidation RMSE: {validation_rmse:.3f}")
print(f"Validation MAE:  {validation_mae:.3f}")
print(
    "Final training RMSE: "
    f"{model.history[-1]['train_rmse']:.3f}"
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 5. Refit on All Observations and Recommend Unseen Items
#
# After model choices have been evaluated, refit the same configuration on all
# observed ratings. Recommendation candidates are the user's genuinely unseen
# items in the original matrix. Already-rated items are excluded before ranking.

# CELL ********************

def recommend_unseen(model, user, observed_ratings, k=3):
    """Return up to k model-ranked items that the known user has not rated."""
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    if user not in observed_ratings.index:
        raise KeyError(
            f"Unknown user {user!r}; use a cold-start fallback or onboarding."
        )
    if user not in model.user_to_pos:
        raise KeyError(f"The fitted model has no factors for user {user!r}.")

    unseen_items = observed_ratings.columns[observed_ratings.loc[user].isna()]
    if len(unseen_items) == 0:
        raise ValueError(f"{user!r} has no unseen items to recommend.")

    scored = []
    for item in unseen_items:
        if item not in model.item_to_pos:
            print(f"Skipping cold-start item without learned factors: {item}")
            continue
        scored.append((item, model.predict(user, item)))

    if not scored:
        raise ValueError(
            f"No unseen item for {user!r} has learned model factors."
        )

    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    selected = scored[:k]
    if len(selected) < k:
        print(
            f"Only {len(selected)} unseen item(s) are available for "
            f"{user!r}; requested {k}."
        )
    return pd.DataFrame(selected, columns=["item", "predicted_rating"])


production_model = BiasedMatrixFactorization(
    n_factors=3,
    learning_rate=0.02,
    regularization=0.05,
    epochs=600,
    seed=31,
).fit(ratings_df)

recommendations = recommend_unseen(
    production_model, "Bob", ratings_df, k=3
)
assert not recommendations["item"].isin(
    ratings_df.columns[ratings_df.loc["Bob"].notna()]
).any()

print("Matrix-factorization recommendations for Bob:")
print(recommendations.round(3).to_string(index=False))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Summary and Limitations
#
# - The model optimizes a stated objective on observed training ratings only.
# - A seeded split and seeded SGD make this small example reproducible.
# - \(L_2\) regularization controls the biases and latent factors.
# - RMSE and MAE are calculated on ratings hidden before training.
# - Recommendations rank only items unseen by the target user.
# - Empty users/items, unknown labels, and no-unseen-item cases produce explicit
#   errors or messages.
#
# This small explicit-feedback example is designed for concepts, not production
# scale. Real systems also tune hyperparameters, compare against simple
# baselines, evaluate ranking and business metrics, monitor drift, and combine
# interaction data with side information for cold-start handling.
