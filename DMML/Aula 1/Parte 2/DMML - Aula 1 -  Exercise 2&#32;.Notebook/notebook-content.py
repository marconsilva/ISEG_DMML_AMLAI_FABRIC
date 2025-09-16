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
# META       "environmentId": "be6440c5-d6f3-4252-85b9-2cfd4b258ea1",
# META       "workspaceId": "03f3982f-785f-4a2f-8ec0-4be54060ee7b"
# META     }
# META   }
# META }

# MARKDOWN ********************

# ## Recap
# So far, you have loaded your data and reviewed it with the following code. Run this cell to set up your coding environment where the previous step left off.

# CELL ********************

%pip install /lakehouse/default/Files/Env/learntools-0.3.4-py2.py3-none-any.whl

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Code you have previously used to load data
import pandas as pd

# Path of the file to read
iowa_file_path = '/lakehouse/default/Files/DMML_Aula1/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *

print("Setup Complete")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Exercises
# 
# ## Step 1: Specify Prediction Target
# Select the target variable, which corresponds to the sales price. Save this to a new variable called `y`. You'll need to print a list of the columns to find the name of the column you need.


# CELL ********************

# print the list of columns in the dataset to find the name of the prediction target


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

y = ____

# Check your answer
step_1.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# The lines below will show you a hint or the solution.
#step_1.hint() 
#step_1.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Step 2: Create X
# Now you will create a DataFrame called `X` holding the predictive features.
# 
# Since you want only some columns from the original data, you'll first create a list with the names of the columns you want in `X`.
# 
# You'll use just the following columns in the list (you can copy and paste the whole list to save some typing, though you'll still need to add quotes):
#   * LotArea
#   * YearBuilt
#   * 1stFlrSF
#   * 2ndFlrSF
#   * FullBath
#   * BedroomAbvGr
#   * TotRmsAbvGrd
# 
# After you've created that list of features, use it to create the DataFrame that you'll use to fit the model.

# CELL ********************

# Create the list of features below
feature_names = ___

# Select data corresponding to features in feature_names
X = ____

# Check your answer
step_2.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

#step_2.hint()
#step_2.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Review Data
# Before building a model, take a quick look at **X** to verify it looks sensible

# CELL ********************

# Review data
# print description or statistics from X
#print(_)

# print the top few lines
#print(_)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Step 3: Specify and Fit Model
# Create a `DecisionTreeRegressor` and save it iowa_model. Ensure you've done the relevant import from sklearn to run this command.
# 
# Then fit the model you just created using the data in `X` and `y` that you saved above.

# CELL ********************

# from _ import _
#specify the model. 
#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = ____

# Fit the model
____

# Check your answer
step_3.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

#step_3.hint()
#step_3.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Step 4: Make Predictions
# Make predictions with the model's `predict` command using `X` as the data. Save the results to a variable called `predictions`.

# CELL ********************

predictions = ____
print(predictions)

# Check your answer
step_4.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

#step_4.hint()
#step_4.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Think About Your Results
# 
# Use the `head` method to compare the top few predictions to the actual home values (in `y`) for those same homes. Anything surprising?


# CELL ********************

# You can write code in this cell


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# It's natural to ask how accurate the model's predictions will be and how you can improve that. That will be you're next step.
# 

