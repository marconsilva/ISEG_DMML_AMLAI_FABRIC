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

# In this exercise, you'll apply what you learned in the **Character encodings** tutorial.
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
from learntools.data_cleaning.ex4 import *
print("Setup Complete")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Get our environment set up
# 
# The first thing we'll need to do is load in the libraries we'll be using.

# CELL ********************

# modules we'll use
import pandas as pd
import numpy as np

# helpful character encoding module
import charset_normalizer

# set seed for reproducibility
np.random.seed(0)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # 1) What are encodings?
# 
# You're working with a dataset composed of bytes.  Run the code cell below to print a sample entry.

# CELL ********************

sample_entry = b'\xa7A\xa6n'
print(sample_entry)
print('data type:', type(sample_entry))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# You notice that it doesn't use the standard UTF-8 encoding. 
# 
# Use the next code cell to create a variable `new_entry` that changes the encoding from `"big5-tw"` to `"utf-8"`.  `new_entry` should have the bytes datatype.

# CELL ********************

new_entry = ____

# Check your answer
q1.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
q1.hint()
q1.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # 2) Reading in files with encoding problems
# 
# Use the code cell below to read in this file at path `"/lakehouse/default/Files/DMML_Aula3/PoliceKillingsUS.csv"`.  
# 
# Figure out what the correct encoding should be and read in the file to a DataFrame `police_killings`.

# CELL ********************

# TODO: Load in the DataFrame correctly.
police_killings = ____

# Check your answer
q2.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Feel free to use any additional code cells for supplemental work.  To get credit for finishing this question, you'll need to run `q2.check()` and get a result of **Correct**.

# CELL ********************

# (Optional) Use this code cell for any additional work.

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

# 
# # Keep going
# 
# In the next lesson, learn how to [**clean up inconsistent text entries**] in your dataset.
