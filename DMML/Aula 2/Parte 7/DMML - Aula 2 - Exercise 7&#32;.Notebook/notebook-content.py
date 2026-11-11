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
# META       "environmentId": "85eb7531-0beb-4054-9c4a-d10a6e21679e",
# META       "workspaceId": "03f3982f-785f-4a2f-8ec0-4be54060ee7b"
# META     }
# META   }
# META }

# MARKDOWN ********************

# In this exercise, you'll apply what you learned in the **Handling missing values** tutorial.
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

from learntools.core import binder
binder.bind(globals())
from learntools.data_cleaning.ex1 import *
print("Setup Complete")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # 1) Take a first look at the data
# 
# Run the next code cell to load in the libraries and dataset you'll use to complete the exercise.

# CELL ********************

# modules we'll use
import pandas as pd
import numpy as np

# read in all our data
sf_permits = pd.read_csv("/lakehouse/default/Files/DMML_Aula3/Building_Permits.csv")

# set seed for reproducibility
np.random.seed(0) 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Use the code cell below to print the first five rows of the `sf_permits` DataFrame.

# CELL ********************

# TODO: Your code here!


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Does the dataset have any missing values?  Once you have an answer, run the code cell below to get credit for your work.

# CELL ********************

# Check your answer (Run this code cell to receive credit!)
q1.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Line below will give you a hint
q1.hint()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # 2) How many missing data points do we have?
# 
# What percentage of the values in the dataset are missing?  Your answer should be a number between 0 and 100.  (If 1/4 of the values in the dataset are missing, the answer is 25.)

# CELL ********************

# TODO: Your code here!
percent_missing = ____

# Check your answer
q2.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
q2.hint()
q2.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # 3) Figure out why the data is missing
# 
# Look at the columns **"Street Number Suffix"** and **"Zipcode"** from the [San Francisco Building Permits dataset]. Both of these contain missing values. 
# - Which, if either, are missing because they don't exist? 
# - Which, if either, are missing because they weren't recorded?  
# 
# Once you have an answer, run the code cell below.

# CELL ********************

# Check your answer (Run this code cell to receive credit!)
q3.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Line below will give you a hint
q3.hint()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # 4) Drop missing values: rows
# 
# If you removed all of the rows of `sf_permits` with missing values, how many rows are left?
# 
# **Note**: Do not change the value of `sf_permits` when checking this.  

# CELL ********************

# TODO: Your code here!


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Once you have an answer, run the code cell below.

# CELL ********************

# Check your answer (Run this code cell to receive credit!)
q4.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Line below will give you a hint
q4.hint()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # 5) Drop missing values: columns
# 
# Now try removing all the columns with empty values.  
# - Create a new DataFrame called `sf_permits_with_na_dropped` that has all of the columns with empty values removed.  
# - How many columns were removed from the original `sf_permits` DataFrame? Use this number to set the value of the `dropped_columns` variable below.

# CELL ********************

# TODO: Your code here
sf_permits_with_na_dropped = ____

dropped_columns = ____

# Check your answer
q5.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
q5.hint()
q5.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # 6) Fill in missing values automatically
# 
# Try replacing all the NaN's in the `sf_permits` data with the one that comes directly after it and then replacing any remaining NaN's with 0.  Set the result to a new DataFrame `sf_permits_with_na_imputed`.

# CELL ********************

# TODO: Your code here
sf_permits_with_na_imputed = ____

# Check your answer
q6.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
q6.hint()
q6.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# 
# 
# In the next lesson, continue learning on data cleaning and transform your data.
