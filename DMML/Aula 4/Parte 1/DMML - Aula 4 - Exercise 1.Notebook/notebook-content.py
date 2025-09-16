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

# This exercise will test your ability to work with data and Logistic Regression.
# 
# We will apply techniques to filter the data, and build a machine learning model with Logistic regression.
# 
# We will work with advertising dataset.
# 
# # Exercises
# 
# Run the following cell to set up code-checking, which will verify your work as you go.

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
from learntools.log_reg.ex1 import *
print("Setup Complete")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Get the Data
# 
# Read in the advertising.csv file and set it to a data frame called ad_data.**

# CELL ********************

ad_data = pd.read_csv('advertising.csv')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Check the head of ad_data

# CELL ********************

ad_data.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ** Use info and describe() on ad_data**

# CELL ********************

ad_data.info()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

ad_data.describe()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Exploratory Data Analysis
# 
# Let's use seaborn to explore the data!
# 
# Try recreating the plots shown below!
# 
# First we Create a histogram of the Age

# CELL ********************

sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Then we Create a jointplot showing Area Income versus Age.

# CELL ********************

sns.jointplot(x='Age',y='Area Income',data=ad_data)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# We then Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.

# CELL ********************

sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,color='red',kind='kde');

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# We also create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**

# CELL ********************

sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Finally, we create a pairplot with the hue defined by the 'Clicked on Ad' column feature.

# CELL ********************

sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Step 1: Logistic Regression Split train and test
# 
# Now it's time to do a train test split!
# 
# Split the data into training set and testing set using train_test_split
# 
# Choose test_size = 0.33 and random_state = 42

# CELL ********************

from sklearn.model_selection import train_test_split

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.33, random_state=42)

step_1.check()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

train_X, val_X, train_y, val_y = _______

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

# # Step 2: Train Logistic Regression Model
# 
# Train and fit a logistic regression model on the training set, again use random_state=42

# CELL ********************

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42)
model.fit(train_X,train_y)

step_2.check()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

from sklearn.linear_model import LogisticRegression
model = _______
#Train here the model
_____

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

# ## Step 3: Make Predictions
# Now predict values for the testing data.

# CELL ********************

val_predictions = model.predict(val_X)
step_3.check()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

val_predictions = __________
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

# Now we can create a classification report for the model.

# CELL ********************

from sklearn.metrics import classification_report

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

print(classification_report(val_y,val_predictions))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Great Job!
# Move on to the next module
