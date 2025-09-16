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
# META     }
# META   }
# META }

# MARKDOWN ********************

# # Introduction #
# 
# In this exercise, you'll apply grid and random search in the [*Ames*] dataset.
# 
# Run this cell to set everything up!

# CELL ********************

%pip install /lakehouse/default/Files/Env/learntools-0.3.4-py2.py3-none-any.whl

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.hyperpar.ex6 import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# we use the random forest regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Pre process of the data
# 
# We will start by reading the data, select some features and split into training and testing

# CELL ********************

# Load the data
train_data = pd.read_csv('/lakehouse/default/Files/DMML_Aula6/home-data-for-ml-course/train.csv')
test_data  = pd.read_csv('/lakehouse/default/Files/DMML_Aula6/home-data-for-ml-course/test.csv')

# Selecting features

features = ['OverallQual', 'GrLivArea', 'GarageCars',  'TotalBsmtSF']

# Selecting the target

X_train       = train_data[features]
y_train       = train_data["SalePrice"]
final_X_test  = test_data[features]

# Split the data into training and validation data
X_train      = X_train.fillna(X_train.mean())
final_X_test = final_X_test.fillna(final_X_test.mean())


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Step 1: Model Creation and Apply Grid Search
# 
# Okay, now you will create a RandomForestRegressor with random_state=0 and we will provide you with a hyperparam_grid which is a grid with hyperparameters, you should use this grid for making an experiment using Grid search, using cross validation with 2 folds.
# In the output you must have the variables n_estimators and max_depth

# CELL ********************

regressor = ______

hyperparam_grid={'max_depth'   : [ 2,  5,  7, 10],
            'n_estimators': [20, 30, 50, 75]}

# your code here - set up the grid search and fit it


the_best_parameters = ______

max_depth = ________
n_estimators = _____________


step_1.check()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
step_1.hint()
step_1.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# This is how you can use the best parameters to fit the data and make new predictions

# CELL ********************



regressor = RandomForestRegressor(
                     n_estimators = the_best_parameters["n_estimators"],
                     max_depth    = the_best_parameters["max_depth"])
regressor.fit(X_train, y_train)

predictions = regressor.predict(final_X_test)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Step 2: Model Creation and Apply Random Search
# 
# Okay, now you will create a RandomForestRegressor with random_state=0 and we will provide you with a hyperparam_grid which is a grid with hyperparameters, you should use this grid for making an experiment using Random search, using cross validation with 2 folds.
# In the output you must have the variables n_estimators and max_depth

# CELL ********************

regressor = ______


hyperparam_grid={'max_depth'   : [ 2,  5,  7, 10],
            'n_estimators': [20, 30, 50, 75]}

# your code here - set up the grid search and fit it


the_best_parameters = ______

max_depth = ________
n_estimators = _____________

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
