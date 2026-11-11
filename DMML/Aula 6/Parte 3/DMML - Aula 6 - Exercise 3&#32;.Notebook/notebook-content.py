# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "jupyter",
# META     "jupyter_kernel_name": "python3.11"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "81cbac54-cfa3-495b-ad48-b44a92bb72fb",
# META       "default_lakehouse_name": "DataScienceLearnLakehouse",
# META       "default_lakehouse_workspace_id": "a677a3bf-5fb2-455e-abaa-9e850bde3e1a",
# META       "known_lakehouses": [
# META         {
# META           "id": "81cbac54-cfa3-495b-ad48-b44a92bb72fb"
# META         }
# META       ]
# META     },
# META     "environment": {
# META       "environmentId": "2b9c63f7-1498-40e2-81b9-a8ccb1b5f193",
# META       "workspaceId": "03f3982f-785f-4a2f-8ec0-4be54060ee7b"
# META     }
# META   }
# META }

# MARKDOWN ********************

# In this exercise, you will use your new knowledge to train a model with **gradient boosting**.
# 
# # Setup
# 
# The questions below will give you feedback on your work. Run the following cell to set up the feedback system.

# CELL ********************

# Set up code checking
import mlflow
mlflow.autolog(disable=True)

import os
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex6 import *
print("Setup Complete")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# You will work with the [Housing Prices ] dataset from the previous exercise. 
# 
# ![Ames Housing dataset image](https://storage.googleapis.com/kaggle-media/learn/images/lTJVG4e.png)
# 
# Run the next code cell without changes to load the training and validation sets in `X_train`, `X_valid`, `y_train`, and `y_valid`.  The test set is loaded in `X_test`.

# CELL ********************

import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X = pd.read_csv('/lakehouse/default/Files/DMML_Aula7/home-data-for-ml-course/train.csv', index_col='Id')
X_test_full = pd.read_csv('/lakehouse/default/Files/DMML_Aula7/home-data-for-ml-course/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)


# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Step 1: Build model
# 
# ### Part A
# 
# In this step, you'll build and train your first model with gradient boosting.
# 
# - Begin by setting `my_model_1` to an XGBoost model.  Use the [XGBRegressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor) class, and set the random seed to 0 (`random_state=0`).  **Leave all other parameters as default.**
# - Then, fit the model to the training data in `X_train` and `y_train`.

# CELL ********************

from xgboost import XGBRegressor

# Define the model
my_model_1 = ____ # Your code here

# Fit the model
____ # Your code here

# Check your answer
step_1.a.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
step_1.a.hint()
step_1.a.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### Part B
# 
# Set `predictions_1` to the model's predictions for the validation data.  Recall that the validation features are stored in `X_valid`.

# CELL ********************

from sklearn.metrics import mean_absolute_error

# Get predictions
predictions_1 = ____ # Your code here

# Check your answer
step_1.b.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
step_1.b.hint()
step_1.b.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### Part C
# 
# Finally, use the `mean_absolute_error()` function to calculate the mean absolute error (MAE) corresponding to the predictions for the validation set.  Recall that the labels for the validation data are stored in `y_valid`.

# CELL ********************

# Calculate MAE
mae_1 = ____ # Your code here

# Uncomment to print MAE
# print("Mean Absolute Error:" , mae_1)

# Check your answer
step_1.c.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
step_1.c.hint()
step_1.c.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Step 2: Improve the model
# 
# Now that you've trained a default model as baseline, it's time to tinker with the parameters, to see if you can get better performance!
# - Begin by setting `my_model_2` to an XGBoost model, using the [XGBRegressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor) class.  Use what you learned in the previous tutorial to figure out how to change the default parameters (like `n_estimators` and `learning_rate`) to get better results.
# - Then, fit the model to the training data in `X_train` and `y_train`.
# - Set `predictions_2` to the model's predictions for the validation data.  Recall that the validation features are stored in `X_valid`.
# - Finally, use the `mean_absolute_error()` function to calculate the mean absolute error (MAE) corresponding to the predictions on the validation set.  Recall that the labels for the validation data are stored in `y_valid`.
# 
# In order for this step to be marked correct, your model in `my_model_2` must attain lower MAE than the model in `my_model_1`. 


# CELL ********************

# Define the model
my_model_2 = ____ # Your code here

# Fit the model
____ # Your code here

# Get predictions
predictions_2 = ____ # Your code here

# Calculate MAE
mae_2 = ____ # Your code here

# Uncomment to print MAE
# print("Mean Absolute Error:" , mae_2)

# Check your answer
step_2.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
step_2.hint()
step_2.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Step 3: Break the model
# 
# In this step, you will create a model that performs worse than the original model in Step 1.  This will help you to develop your intuition for how to set parameters.  You might even find that you accidentally get better performance, which is ultimately a nice problem to have and a valuable learning experience!
# - Begin by setting `my_model_3` to an XGBoost model, using the [XGBRegressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor) class.  Use what you learned in the previous tutorial to figure out how to change the default parameters (like `n_estimators` and `learning_rate`) to design a model to get high MAE.
# - Then, fit the model to the training data in `X_train` and `y_train`.
# - Set `predictions_3` to the model's predictions for the validation data.  Recall that the validation features are stored in `X_valid`.
# - Finally, use the `mean_absolute_error()` function to calculate the mean absolute error (MAE) corresponding to the predictions on the validation set.  Recall that the labels for the validation data are stored in `y_valid`.
# 
# In order for this step to be marked correct, your model in `my_model_3` must attain higher MAE than the model in `my_model_1`. 


# CELL ********************

# Define the model
my_model_3 = ____

# Fit the model
____ # Your code here

# Get predictions
predictions_3 = ____

# Calculate MAE
mae_3 = ____

# Uncomment to print MAE
# print("Mean Absolute Error:" , mae_3)

# Check your answer
step_3.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
step_3.hint()
step_3.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Congratulations
# 
# You have finished the learning on Trees and Ensemble methods
