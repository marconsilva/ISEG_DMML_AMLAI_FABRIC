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

# In this exercise, you will use your new knowledge to propose a solution to a real-world scenario.  To succeed, you will need to import data into Python, answer questions using the data, and generate **histograms** and **density plots** to understand patterns in the data.
# 
# ## Scenario
# 
# You'll work with a real-world dataset containing information collected from microscopic images of breast cancer tumors, similar to the image below.
# 
# ![ex4_cancer_image](https://storage.googleapis.com/kaggle-media/learn/images/qUESsJe.png)
# 
# Each tumor has been labeled as either [**benign**](https://en.wikipedia.org/wiki/Benign_tumor) (_noncancerous_) or **malignant** (_cancerous_).
# 
# To learn more about how this kind of data is used to create intelligent algorithms to classify tumors in medical settings, **watch the short video as one example [at this link](https://www.youtube.com/watch?v=9Mz84cwVmS0)**.

# MARKDOWN ********************

# 
# 
# ## Setup
# 
# Run the next cell to import and configure the Python libraries that you need to complete the exercise.

# CELL ********************

%pip install /lakehouse/default/Files/Env/learntools-0.3.4-py2.py3-none-any.whl
%pip install seaborn

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# The questions below will give you feedback on your work. Run the following cell to set up our feedback system.

# CELL ********************

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex5 import *
print("Setup Complete")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Step 1: Load the data
# 
# In this step, you will load the data.
# - Load the data file into a DataFrame called `cancer_data`.  
# - The corresponding filepath is `cancer_filepath`.  
# - Use the `"Id"` column to label the rows.

# CELL ********************

# Path of the files to read
cancer_filepath = "/lakehouse/default/Files/DMML_Aula2/cancer.csv"

# Fill in the line below to read the file into a variable cancer_data
cancer_data = ____

# Run the line below with no changes to check that you've loaded the data correctly
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

# ## Step 2: Review the data
# 
# Use a Python command to print the first 5 rows of the data.

# CELL ********************

# Print the first five rows of the data
____ # Your code here

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# In the dataset, each row corresponds to a different image.  The dataset has 31 different columns, corresponding to:
# - 1 column (`'Diagnosis'`) that classifies tumors as either benign (which appears in the dataset as **`B`**) or malignant (__`M`__), and
# - 30 columns containing different measurements collected from the images.
# 
# Use the first 5 rows of the data to answer the questions below.

# CELL ********************

# Fill in the line below: In the first five rows of the data, what is the
# largest value for 'Perimeter (mean)'?
max_perim = ____

# Fill in the line below: What is the value for 'Radius (mean)' for the tumor with Id 8510824?
mean_radius = ____

# Check your answers
step_2.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Lines below will give you a hint or solution code, uncoment for that
#step_2.hint()
#step_2.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Step 3: Investigating differences

# MARKDOWN ********************

# #### Part A
# 
# Use the code cell below to create two histograms that show the distribution in values for `'Area (mean)'`, separately for both benign and malignant tumors.  (_To permit easy comparison, create a single figure containing both histograms in the code cell below._)

# CELL ********************

# Histograms for benign and maligant tumors
____

# Check your answer
step_3.a.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
step_3.a.hint()
step_3.a.solution_plot()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# #### Part B
# 
# A researcher approaches you for help with identifying how the `'Area (mean)'` column can be used to understand the difference between benign and malignant tumors.  Based on the histograms above, 
# - Do malignant tumors have higher or lower values for `'Area (mean)'` (relative to benign tumors), on average?
# - Which tumor type seems to have a larger range of potential values?

# CELL ********************

#step_3.b.hint()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Check your answer 
step_3.b.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Step 4: A very useful column
# 
# #### Part A
# 
# Use the code cell below to create two KDE plots that show the distribution in values for `'Radius (worst)'`, separately for both benign and malignant tumors.  (_To permit easy comparison, create a single figure containing both KDE plots in the code cell below._)

# CELL ********************

# KDE plots for benign and malignant tumors
____ 

# Check your answer
step_4.a.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
#step_4.a.hint()
#step_4.a.solution_plot()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# #### Part B
# 
# A hospital has recently started using an algorithm that can diagnose tumors with high accuracy.  Given a tumor with a value for `'Radius (worst)'` of 25, do you think the algorithm is more likely to classify the tumor as benign or malignant?

# CELL ********************

#_COMMENT_IF(PROD)_
step_4.b.hint()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Check your answer (Run this code cell to receive credit!)
step_4.b.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Keep going
# 
# Review all that you've learned and explore how to further customize your plots in the **[next tutorial]**!
