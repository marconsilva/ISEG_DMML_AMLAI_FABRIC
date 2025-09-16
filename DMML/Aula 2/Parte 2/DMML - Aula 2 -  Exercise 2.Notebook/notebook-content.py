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

# In this exercise, you will use your new knowledge to propose a solution to a real-world scenario. To succeed, you will need to import data into Python, answer questions using the data, and generate **line charts** to understand patterns in the data.
# 
# ## Scenario
# 
# You have recently been hired to manage the museums in the City of Los Angeles. Your first project focuses on the four museums pictured in the images below.
# 
# ![ex1_museums](https://storage.googleapis.com/kaggle-media/learn/images/pFYL8J1.png)
# 
# You will leverage data from the Los Angeles [Data Portal](https://data.lacity.org/) that tracks monthly visitors to each museum.  
# 
# ![ex1_xlsx](https://storage.googleapis.com/kaggle-media/learn/images/mGWYlym.png)
# 
# ## Setup
# 
# Run the next cell to import and configure the Python libraries that you need to complete the exercise.

# CELL ********************

%pip install /lakehouse/default/Files/Env/learntools-0.3.4-py2.py3-none-any.whl

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

# The questions below will give you feedback on your work. Run the following cell to set up the feedback system.

# CELL ********************

# Set up code checking
import os
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex2 import *
print("Setup Complete")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Step 1: Load the data
# 
# Your first assignment is to read the LA Museum Visitors data file into `museum_data`.  Note that:
# - The filepath to the dataset is stored as `museum_filepath`.  Please **do not** change the provided value of the filepath.
# - The name of the column to use as row labels is `"Date"`.  (This can be seen in cell A1 when the file is opened in Excel.)
# 
# To help with this, you may find it useful to revisit some relevant code from the tutorial, which we have pasted below:
# 
# ```python
# # Path of the file to read
# spotify_filepath = "/lakehouse/default/Files/DMML_Aula2/spotify.csv"
# 
# # Read the file into a variable spotify_data
# spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)
# ```
# 
# The code you need to write now looks very similar!

# CELL ********************

# Path of the file to read
museum_filepath = "/lakehouse/default/Files/DMML_Aula2/museum_visitors.csv"

# Fill in the line below to read the file into a variable museum_data
museum_data = ____

# Run the line below with no changes to check that you've loaded the data correctly
step_1.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Uncomment the line below to receive a hint
#step_1.hint()
# Uncomment the line below to see the solution
#step_1.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Step 2: Review the data
# 
# Use a Python command to print the last 5 rows of the data.

# CELL ********************

# Print the last five rows of the data 
____ # Your code here

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# The last row (for `2018-11-01`) tracks the number of visitors to each museum in November 2018, the next-to-last row (for `2018-10-01`) tracks the number of visitors to each museum in October 2018, _and so on_.
# 
# Use the last 5 rows of the data to answer the questions below.

# CELL ********************

# Fill in the line below: How many visitors did the Chinese American Museum 
# receive in July 2018?
ca_museum_jul18 = ____ 

# Fill in the line below: In October 2018, how many more visitors did Avila 
# Adobe receive than the Firehouse Museum?
avila_oct18 = ____

# Check your answers
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

# ## Step 3: Convince the museum board 
# 
# The Firehouse Museum claims they ran an event in 2014 that brought an incredible number of visitors, and that they should get extra budget to run a similar event again.  The other museums think these types of events aren't that important, and budgets should be split purely based on recent visitors on an average day.  
# 
# To show the museum board how the event compared to regular traffic at each museum, create a line chart that shows how the number of visitors to each museum evolved over time.  Your figure should have four lines (one for each museum).
# 
# > **(Optional) Note**: If you have some prior experience with plotting figures in Python, you might be familiar with the `plt.show()` command.  If you decide to use this command, please place it **after** the line of code that checks your answer (in this case, place it after `step_3.check()` below) -- otherwise, the checking code will return an error!

# CELL ********************

# Line chart showing the number of visitors to each museum over time
____ # Your code here

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
step_3.solution_plot()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Step 4: Assess seasonality
# 
# When meeting with the employees at Avila Adobe, you hear that one major pain point is that the number of museum visitors varies greatly with the seasons, with low seasons (when the employees are perfectly staffed and happy) and also high seasons (when the employees are understaffed and stressed).  You realize that if you can predict these high and low seasons, you can plan ahead to hire some additional seasonal employees to help out with the extra work.
# 
# #### Part A
# Create a line chart that shows how the number of visitors to Avila Adobe has evolved over time.  (_If your code returns an error, the first thing that you should check is that you've spelled the name of the column correctly!  You must write the name of the column exactly as it appears in the dataset._)

# CELL ********************

# Line plot showing the number of visitors to Avila Adobe over time
____ # Your code here

# Check your answer
step_4.a.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
step_4.a.hint()
step_4.a.solution_plot()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# #### Part B
# 
# Does Avila Adobe get more visitors:
# - in September-February (in LA, the fall and winter months), or 
# - in March-August (in LA, the spring and summer)?  
# 
# Using this information, when should the museum staff additional seasonal employees?

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

# # Keep going
# 
# Move on to learn about **[bar charts and heatmaps]** with a new dataset!
