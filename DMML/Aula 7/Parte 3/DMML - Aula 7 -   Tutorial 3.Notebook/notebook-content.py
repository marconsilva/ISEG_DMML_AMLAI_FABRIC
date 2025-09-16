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

# In this tutorial, you will learn how to build and optimize models with **gradient boosting**.  This method is widely used and achieves state-of-the-art results on a variety of datasets. 
# 
# # Introduction
# 
# For much of this course, you have made predictions with the random forest method, which achieves better performance than a single decision tree simply by averaging the predictions of many decision trees.
# 
# We refer to the random forest method as an "ensemble method" and specifically bagging.  By definition, **ensemble methods** combine the predictions of several models (e.g., several trees, in the case of random forests).  
# 
# Next, we'll learn about another ensemble method called gradient boosting.  
# 
# # Gradient Boosting
# 
# **Gradient boosting** is a method that goes through cycles to iteratively add models into an ensemble.  
# 
# It begins by initializing the ensemble with a single model, whose predictions can be pretty naive.  (Even if its predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors.)
# 
# Then, we start the cycle:
# - First, we use the current ensemble to generate predictions for each observation in the dataset.  To make a prediction, we add the predictions from all models in the ensemble.  
# - These predictions are used to calculate a loss function (like [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error), for instance).
# - Then, we use the loss function to fit a new model that will be added to the ensemble.  Specifically, we determine model parameters so that adding this new model to the ensemble will reduce the loss.  (*Side note: The "gradient" in "gradient boosting" refers to the fact that we'll use [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) on the loss function to determine the parameters in this new model.*)
# - Finally, we add the new model to ensemble, and ...
# - ... repeat!
# 
# ![tut6_boosting](https://storage.googleapis.com/kaggle-media/learn/images/MvCGENh.png)
# 
# 
# # Example
# 
# We begin by loading the training and validation data in `X_train`, `X_valid`, `y_train`, and `y_valid`. 


# CELL ********************

import mlflow
mlflow.autolog(disable=True)
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('/lakehouse/default/Files/DMML_Aula7/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# In this example, you'll work with the XGBoost library.  **XGBoost** stands for **extreme gradient boosting**, which is an implementation of gradient boosting with several additional features focused on performance and speed.  (_Scikit-learn has another version of gradient boosting, but XGBoost has some technical advantages._) 
# 
# In the next code cell, we import the scikit-learn API for XGBoost ([`xgboost.XGBRegressor`](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn)).  This allows us to build and fit a model just as we would in scikit-learn.  As you'll see in the output, the `XGBRegressor` class has many tunable parameters -- you'll learn about those soon!

# CELL ********************

from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(X_train, y_train)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# We also make predictions and evaluate the model.

# CELL ********************

from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Parameter Tuning
# 
# XGBoost has a few parameters that can dramatically affect accuracy and training speed.  The first parameters you should understand are:
# 
# ### `n_estimators`
# `n_estimators` specifies how many times to go through the modeling cycle described above.  It is equal to the number of models that we include in the ensemble. 
# 
# - Too _low_ a value causes _underfitting_, which leads to inaccurate predictions on both training data and test data. 
# - Too _high_ a value causes _overfitting_, which causes accurate predictions on training data, but inaccurate predictions on test data (_which is what we care about_). 
# 
# Typical values range from 100-1000, though this depends a lot on the `learning_rate` parameter discussed below.
# 
# Here is the code to set the number of models in the ensemble:

# CELL ********************

my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### `early_stopping_rounds`
# 
# `early_stopping_rounds` offers a way to automatically find the ideal value for `n_estimators`. Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for `n_estimators`.  It's smart to set a high value for `n_estimators` and then use `early_stopping_rounds` to find the optimal time to stop iterating.
# 
# Since random chance sometimes causes a single round where validation scores don't improve, you need to specify a number for how many rounds of straight deterioration to allow before stopping.  Setting `early_stopping_rounds=5` is a reasonable choice.  In this case, we stop after 5 straight rounds of deteriorating validation scores.
# 
# When using `early_stopping_rounds`, you also need to set aside some data for calculating the validation scores - this is done by setting the `eval_set` parameter.  
# 
# We can modify the example above to include early stopping:

# CELL ********************

my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)],
             verbose=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# If you later want to fit a model with all of your data, set `n_estimators` to whatever value you found to be optimal when run with early stopping.
# 
# ### `learning_rate`
# 
# Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number (known as the **learning rate**) before adding them in.  
# 
# This means each tree we add to the ensemble helps us less.  So, we can set a higher value for `n_estimators` without overfitting.  If we use early stopping, the appropriate number of trees will be determined automatically.
# 
# In general, a small learning rate and large number of estimators will yield more accurate XGBoost models, though it will also take the model longer to train since it does more iterations through the cycle.  As default, XGBoost sets `learning_rate=0.1`.
# 
# Modifying the example above to change the learning rate yields the following code:

# CELL ********************

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### `n_jobs`
# On larger datasets where runtime is a consideration, you can use parallelism to build your models faster.  It's common to set the parameter `n_jobs` equal to the number of cores on your machine.  On smaller datasets, this won't help. 
# 
# The resulting model won't be any better, so micro-optimizing for fitting time is typically nothing but a distraction. But, it's useful in large datasets where you would otherwise spend a long time waiting during the `fit` command.
# 
# Here's the modified example:

# CELL ********************

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Conclusion
# 
# [XGBoost](https://xgboost.readthedocs.io/en/latest/) is a leading software library for working with standard tabular data (the type of data you store in Pandas DataFrames, as opposed to more exotic types of data like images and videos).  With careful parameter tuning, you can train highly accurate models. 
# 
# # Your Turn
# 
# Train your own model with XGBoost in the **[next exercise]**!
