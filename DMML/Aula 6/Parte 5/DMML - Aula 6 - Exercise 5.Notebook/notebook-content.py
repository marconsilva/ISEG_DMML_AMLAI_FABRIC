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
# META       "environmentId": "bbfc78f6-eb02-8b9e-41db-f4d8a067db81",
# META       "workspaceId": "00000000-0000-0000-0000-000000000000"
# META     }
# META   }
# META }

# MARKDOWN ********************

# In this exercise, you will leverage what you've learned to tune a machine learning model with **cross-validation**.
# 
# # Setup
# 
# The questions below will give you feedback on your work. Run the following cell to set up the feedback system.

# CELL ********************

%pip install /lakehouse/default/Files/Env/learntools-0.3.4-py2.py3-none-any.whl

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Set up code checking
import os
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex5 import *
print("Setup Complete")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# You will work with the [Housing Prices dataset] from the previous exercise. 
# 
# ![Ames Housing dataset image](https://storage.googleapis.com/kaggle-media/learn/images/lTJVG4e.png)
# 
# Run the next code cell without changes to load the training and test data in `X` and `X_test`.  For simplicity, we drop categorical variables.

# CELL ********************

import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
train_data = pd.read_csv('/lakehouse/default/Files/DMML_Aula6/home-data-for-ml-course/train.csv', index_col='Id')
test_data = pd.read_csv('/lakehouse/default/Files/DMML_Aula6/home-data-for-ml-course/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice              
train_data.drop(['SalePrice'], axis=1, inplace=True)

# Select numeric columns only
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Use the next code cell to print the first several rows of the data.

# CELL ********************

X.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# So far, you've learned how to build pipelines with scikit-learn.  For instance, the pipeline below will use [`SimpleImputer()`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) to replace missing values in the data, before using [`Ridge()`](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_coeffs.html) to train a Ridge regression model to make predictions. In this case we are  setting the value of alpha to 1

# CELL ********************

from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# Assuming X and y are already defined

# Step 1: Impute missing values
imputer = SimpleImputer()
X_imputed = imputer.fit_transform(X)

# Generate values for `alpha` that are evenly distributed on a logarithmic scale
alphas = np.logspace(-3, 4, 200)

# Step 2: Define the model
model = Ridge(alpha=1, random_state=42)


# Step 3: Evaluate the model using cross-validation
# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(model, X_imputed, y, cv=5, scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)

print("Average MAE score:", scores.mean())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# You have also learned how to use cross-validation.  The code above uses the [`cross_val_score()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) function to obtain the mean absolute error (MAE), averaged across five different folds.  Recall we set the number of folds with the `cv` parameter.

# MARKDOWN ********************

# # Step 1: Write a useful function
# 
# In this exercise, you'll use cross-validation to select parameters for a machine learning model.
# 
# Begin by writing a function `get_score()` that reports the average (over three cross-validation folds) MAE of a machine learning pipeline that uses:
# - the data in `X` and `y` to create folds,
# - `SimpleImputer()` (with all parameters left as default) to replace missing values, and
# - `Ridge()` (with `random_state=0`) to fit a random forest model.
# - Make sure that the output from the function is using .round(2) ! 
# 
# The `alpha` parameter supplied to `get_score()` is used when setting the ridge regression, alpha which is a positive constant that multiplies the penalty term, controlling the regularization strength. 

# CELL ********************

def get_score(alpha):

    """Return the average MAE over 3 CV folds ofridge  model.
    
    Keyword argument:
    alpha -- positive constant that multiplies the penalty term
    """
    # Replace this body with your own code
    pass

# Check your answer
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

# # Step 2: Test different parameter values
# 
# Now, you will use the function that you defined in Step 1 to evaluate the model performance corresponding to eight different values for the number of alphas in the Ridge: 50, 100, 150, ..., 300, 350, 400. (ideally should be as above in log scale)
# 
# Store your results in a Python dictionary `results`, where `results[i]` is the average MAE returned by `get_score(i)`.

# CELL ********************

x_i = [round(i) for i in list(results.values())]
print(x_i)
assert alpha < 18000

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

results = ____ # Your code here



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

# Use the next cell to visualize your results from Step 2.  Run the code without changes.

# CELL ********************

import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(list(results.keys()), list(results.values()))
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Step 3: Find the best parameter value
# 
# Given the results, which value for `alpha` seems best for the random forest model?  Use your answer to set the value of `alpha_best`.

# CELL ********************

alpha_best = ____

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

# In this exercise, you have explored one method for choosing appropriate parameters in a machine learning model.  
# 
# In some way what you implemeted was [hyperparameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization).
# Thankfully, scikit-learn also contains a built-in function [`GridSearchCV()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) that can make your grid search code very efficient!
# 
# # Keep going
# 

