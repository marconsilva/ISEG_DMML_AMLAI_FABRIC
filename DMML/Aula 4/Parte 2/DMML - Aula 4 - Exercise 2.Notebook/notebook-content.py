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

# # Introduction
# 
# In the exercise, you will work with data from the TalkingData AdTracking.  The goal is to predict if a user will download an app after clicking through an ad. 
# 
# 
# For this course you will use a small sample of the data, dropping 99% of negative records (where the app wasn't downloaded) to make the target more balanced.
# 
# After building a baseline model, you'll be able to see how your feature engineering and selection efforts improve the model's performance.
# 
# ## Setup
# 
# Begin by running the code cell below to set up the exercise.

# CELL ********************

%pip install /lakehouse/default/Files/Env/learntools-0.3.4-py2.py3-none-any.whl

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.feature_engineering.ex1 import *

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Baseline Model
# 
# The first thing you'll do is construct a baseline model. We'll begin by looking at the data.

# CELL ********************

import pandas as pd

click_data = pd.read_csv('/lakehouse/default/Files/DMML_Aula4/feature-engineering-data/train_sample.csv',
                         parse_dates=['click_time'])
click_data.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### 1) Construct features from timestamps
# 
# Notice that the `click_data` DataFrame has a `'click_time'` column with timestamp data.
# 
# Use this column to create features for the coresponding day, hour, minute and second. 
# 
# Store these as new integer columns `day`, `hour`, `minute`, and `second` in a new DataFrame `clicks`.

# CELL ********************

# Add new columns for timestamp features day, hour, minute, and second
clicks = click_data.copy()
clicks['day'] = clicks['click_time'].dt.day.astype('uint8')
# Fill in the rest
clicks['hour'] = ____
clicks['minute'] = ____
clicks['second'] = ____

# Check your answer
q_1.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Uncomment these if you need guidance
q_1.hint()
q_1.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### 2) Label Encoding
# For each of the categorical features `['ip', 'app', 'device', 'os', 'channel']`, use scikit-learn's `LabelEncoder` to create new features in the `clicks` DataFrame. The new column names should be the original column name with `'_labels'` appended, like `ip_labels`.

# CELL ********************

from sklearn import preprocessing

cat_features = ['ip', 'app', 'device', 'os', 'channel']

# Create new columns in clicks using preprocessing.LabelEncoder()
for feature in cat_features:
    ____

# Check your answer
q_2.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Uncomment these if you need guidance
# q_2.hint()
# q_2.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Run the next code cell to view your new DataFrame.

# CELL ********************

clicks.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### 3) One-hot Encoding
# 
# In the code cell above, you used label encoded features.  Would it have also made sense to instead use one-hot encoding for the categorical variables `'ip'`, `'app'`, `'device'`, `'os'`, or `'channel'`?
# 
# **Note**: If you're not familiar with one-hot encoding, please check out aula 4.
# 
# Run the following line after you've decided your answer.

# CELL ********************

# Check your answer (Run this code cell )
q_3.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Keep Going
# Now that you have a baseline idea keep going !
