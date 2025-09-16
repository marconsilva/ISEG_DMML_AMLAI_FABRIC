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

# ## Decision Tree Classifier 
# 
# 
# In this exercise you will build a Decision Tree Classifier to predict the safety of the car. 

# CELL ********************

%pip install /lakehouse/default/Files/Env/learntools-0.3.4-py2.py3-none-any.whl

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Import the libraries and setup


# CELL ********************



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
%matplotlib inline



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Run the cell bellow to complete the setup

# CELL ********************

# Set up code checking
import os
from learntools.core import binder
binder.bind(globals())
from learntools.decision_tree.ex1 import *
print("Setup Complete")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

import warnings

warnings.filterwarnings('ignore')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Import dataset

# CELL ********************

data = '/lakehouse/default/Files/AMLAI_Aula7/car_evaluation.csv'

df = pd.read_csv(data, header=None)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Exploratory data analysis
# 
# 
# Now, we will explore the data to gain insights about the data. 

# CELL ********************

# view dimensions of dataset

df.shape

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# We can see that there are 1728 instances and 7 variables in the data set.

# MARKDOWN ********************

# ### View top 5 rows of dataset

# CELL ********************

# preview the dataset

df.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### Rename column names
# 
# We can see that the dataset does not have proper column names. The columns are merely labelled as 0,1,2.... and so on. We should give proper names to the columns. I will do it as follows:-

# CELL ********************

col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']


df.columns = col_names

col_names

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# let's again preview the dataset

df.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# We can see that the column names are renamed. Now, the columns have meaningful names.

# MARKDOWN ********************

# ### View summary of dataset

# CELL ********************

df.info()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### Frequency distribution of values in variables
# 
# Now, I will check the frequency counts of categorical variables.

# CELL ********************

col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']


for col in col_names:
    
    print(df[col].value_counts())   


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# We can see that the `doors` and `persons` are categorical in nature. So, we will treat them as categorical variables.

# MARKDOWN ********************

# ### Summary of variables
# 
# 
# - There are 7 variables in the dataset. All the variables are of categorical data type.
# 
# 
# - These are given by `buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety` and `class`.
# 
# 
# - `class` is the target variable.

# MARKDOWN ********************

# ### Explore `class` variable

# CELL ********************

df['class'].value_counts()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# The `class` target variable is ordinal in nature.

# MARKDOWN ********************

# ### Missing values in variables

# CELL ********************

# check missing values in variables

df.isnull().sum()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# We can see that there are no missing values in the dataset. I have checked the frequency distribution of values previously. It also confirms that there are no missing values in the dataset.

# MARKDOWN ********************

# ## Declare feature vector and target variable


# CELL ********************

X = df.drop(['class'], axis=1)

y = df['class']

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Step 1: Split data into separate training and test set
# 
# Split the data into train and test with 0.33 test size and random_state=42


# CELL ********************

from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = ___

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

#Lines below will give you a hint or solution code
step_1.hint()
step_1.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# check the shape of X_train and X_test

X_train.shape, X_test.shape

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Feature Engineering
# 
# 
# **Feature Engineering** is the process of transforming raw data into useful features that help us to understand our model better and increase its predictive power. We will carry out feature engineering on different types of variables.
# 
# 
# First, we will check the data types of variables again.

# CELL ********************

# check data types in X_train

X_train.dtypes

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### Encode categorical variables
# 
# 
# Now, we will encode the categorical variables.

# CELL ********************

X_train.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# We can see that all  the variables are ordinal categorical data type.

# CELL ********************

# import category encoders

import category_encoders as ce

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# encode variables with ordinal encoding

encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])


X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

X_train.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

X_test.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# We now have training and test set ready for model building. 

# MARKDOWN ********************

# # Step 2: Create Decision Tree Classifier with criterion entropy
# Let's now create our decision tree with criterion = 'entropy' and max_depth=3, ensure the random_state is 0 and finally fit the model

# CELL ********************

from sklearn.tree import DecisionTreeClassifier

# instantiate the DecisionTreeClassifier model with criterion entropy

#clf_en = ____

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

#lines below will give you a hint or solution code
step_2.hint()
step_2.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### Predict the Test set results with criterion entropy

# CELL ********************

y_pred_en = clf_en.predict(X_test)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### Check accuracy score with criterion entropy

# CELL ********************

from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test, y_pred_en)))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### Compare the train-set and test-set accuracy
# 
# 
# Now, we will compare the train-set and test-set accuracy to check for overfitting.

# CELL ********************

y_pred_train_en = clf_en.predict(X_train)

y_pred_train_en

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_en)))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### Check for overfitting and underfitting

# CELL ********************

# print the scores on training and test set

print('Training set score: {:.4f}'.format(clf_en.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_en.score(X_test, y_test)))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# We can see that the training-set score and test-set score is same as above. The training-set accuracy score is 0.7865 while the test-set accuracy to be 0.8021. These two values are quite comparable. So, there is no sign of overfitting. 


# MARKDOWN ********************

# ### Visualize decision-trees

# CELL ********************

plt.figure(figsize=(12,8))

from sklearn import tree

tree.plot_tree(clf_en.fit(X_train, y_train)) 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### Visualize decision-trees with graphviz

# CELL ********************

import graphviz 
dot_data = tree.export_graphviz(clf_en, out_file=None, 
                              feature_names=X_train.columns,  
                              class_names=y_train,  
                              filled=True, rounded=True,  
                              special_characters=True)

graph = graphviz.Source(dot_data) 

graph 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Now, based on the above analysis we can conclude that our classification model accuracy is very good. Our model is doing a very good job in terms of predicting the class labels.
# 
# 
# But, it does not give the underlying distribution of values. Also, it does not tell anything about the type of errors our classifer is making. 
# 
# 
# We have another tool called `Confusion matrix` that comes to our rescue.

# MARKDOWN ********************

# # Step 3: Calculate Confusion matrix
# 
# Create the confusion matrix

# CELL ********************

from sklearn.metrics import confusion_matrix

#cm = ______

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

#lines below will give you a hint or solution code
step_3.hint()
step_3.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Classification Report
# 
# 
# **Classification report** is another way to evaluate the classification model performance. It displays the  **precision**, **recall**, **f1** and **support** scores for the model. 
# 
# We can print a classification report as follows:-

# CELL ********************

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_en))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Keep going
# 
# You have finished the first part with decision tree, it is time to move to the next one ! 
