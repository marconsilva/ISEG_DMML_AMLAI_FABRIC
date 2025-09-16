# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "a5591839-f387-4a67-a52e-dac9b3ea21b0",
# META       "default_lakehouse_name": "DataScienceLearnLakehouse",
# META       "default_lakehouse_workspace_id": "03f3982f-785f-4a2f-8ec0-4be54060ee7b"
# META     },
# META     "environment": {
# META       "environmentId": "2b9c63f7-1498-40e2-81b9-a8ccb1b5f193",
# META       "workspaceId": "03f3982f-785f-4a2f-8ec0-4be54060ee7b"
# META     }
# META   }
# META }

# MARKDOWN ********************

# # Introduction #
# 
# In this exercise you'll start developing features. As you work through this exercise, you might take a moment to look at the data documentation again and consider whether the features we're creating make sense from a real-world perspective, and whether there are any useful combinations that stand out to you.
# 
# Run this cell to set everything up!

# CELL ********************

# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.feature_engineering_new.ex3 import *

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_log_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score


# Prepare data
df = pd.read_csv("/lakehouse/default/Files/DMML_Aula4/fe-course-data/ames.csv")
X = df.copy()
y = X.pop("SalePrice")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# -------------------------------------------------------------------------------
# 
# Let's start with a few mathematical combinations. We'll focus on features describing areas -- having the same units (square-feet) makes it easy to combine them in sensible ways. Since we're using XGBoost (a tree-based model), we'll focus on ratios and sums.
# 
# # 1) Create Mathematical Transforms
# 
# Create the following features:
# 
# - `LivLotRatio`: the ratio of `GrLivArea` to `LotArea`
# - `Spaciousness`: the sum of `FirstFlrSF` and `SecondFlrSF` divided by `TotRmsAbvGrd`
# - `TotalOutsideSF`: the sum of `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `Threeseasonporch`, and `ScreenPorch`

# CELL ********************

# YOUR CODE HERE
X_1 = pd.DataFrame()  # dataframe to hold new features

X_1["LivLotRatio"] = ____
X_1["Spaciousness"] = ____
X_1["TotalOutsideSF"] = ____


# Check your answer
q_1.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
#q_1.hint()
#q_1.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# -------------------------------------------------------------------------------
# 
# If you've discovered an interaction effect between a numeric feature and a categorical feature, you might want to model it explicitly using a one-hot encoding, like so:
# 
# ```
# # One-hot encode Categorical feature, adding a column prefix "Cat"
# X_new = pd.get_dummies(df.Categorical, prefix="Cat")
# 
# # Multiply row-by-row
# X_new = X_new.mul(df.Continuous, axis=0)
# 
# # Join the new features to the feature set
# X = X.join(X_new)
# ```
# 
# # 2) Interaction with a Categorical
# 
# Use `BldgType` and `GrLivArea` in create their interaction features.

# CELL ********************

# YOUR CODE HERE
# One-hot encode BldgType. Use `prefix="Bldg"` in `get_dummies`
X_2 = ____ 
# Multiply
X_2 = ____


# Check your answer
q_2.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
q_2.hint()
q_2.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # 3) Count Feature
# 
# Let's try creating a feature that describes how many kinds of outdoor areas a dwelling has. Create a feature `PorchTypes` that counts how many of the following are greater than 0.0:
# 
# ```
# WoodDeckSF
# OpenPorchSF
# EnclosedPorch
# Threeseasonporch
# ScreenPorch
# ```

# CELL ********************

X_3 = pd.DataFrame()

X_3["PorchTypes"] = ____


# Check your answer
q_3.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
#q_3.hint()
#q_3.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # 4) Break Down a Categorical Feature
# 
# `MSSubClass` describes the type of a dwelling:

# CELL ********************

df.MSSubClass.unique()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# You can see that there is a more general categorization described (roughly) by the first word of each category. Create a feature containing only these first words by splitting `MSSubClass` at the first underscore `_`. (Hint: In the `split` method use an argument `n=1`.)

# CELL ********************

X_4 = pd.DataFrame()

# YOUR CODE HERE
____

# Check your answer
q_4.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
#q_4.hint()
#q_4.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # 5) Use a Grouped Transform
# 
# The value of a home often depends on how it compares to typical homes in its neighborhood. Create a feature `MedNhbdArea` that describes the *median* of `GrLivArea` grouped on `Neighborhood`.

# CELL ********************

X_5 = pd.DataFrame()

# YOUR CODE HERE
X_5["MedNhbdArea"] = ____

# Check your answer
q_5.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
#q_5.hint()
#q_5.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Now you've made your first new feature set! If you like, you can run the cell below to score the model with all of your new features added:

# CELL ********************

X_new = X.join([X_1, X_2, X_3, X_4, X_5])
score_dataset(X_new, y)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Keep Going #
# 

