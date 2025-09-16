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

# # Introduction
# 
# In this tutorial, you'll learn all about **[pandas](https://pandas.pydata.org)**, the most popular Python library for data analysis.
# 
# Along the way, you'll complete several hands-on exercises with real-world data.  We recommend that you work on the exercises while reading the corresponding tutorials.  
# 
# You will always have a tutorial and then the exercise corresponding
# 
# In this tutorial, you will learn how to create your own data, along with how to work with data that already exists.  
# 
# # Getting started
# 
# To use pandas, you'll typically start with the following line of code.

# CELL ********************

import pandas as pd 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Creating data
# 
# There are two core objects in pandas: the **DataFrame** and the **Series**.
# 
# ### DataFrame
# 
# A DataFrame is a table. It contains an array of individual *entries*, each of which has a certain *value*. Each entry corresponds to a row (or *record*) and a *column*.
# 
# For example, consider the following simple DataFrame:

# CELL ********************

pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# In this example, the "0, No" entry has the value of 131. The "0, Yes" entry has a value of 50, and so on.
# 
# DataFrame entries are not limited to integers. For instance, here's a DataFrame whose values are strings:

# CELL ********************

pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# We are using the `pd.DataFrame()` constructor to generate these DataFrame objects. The syntax for declaring a new one is a dictionary whose keys are the column names (`Bob` and `Sue` in this example), and whose values are a list of entries. This is the standard way of constructing a new DataFrame, and the one you are most likely to encounter.

# MARKDOWN ********************

# The dictionary-list constructor assigns values to the *column labels*, but just uses an ascending count from 0 (0, 1, 2, 3, ...) for the *row labels*. Sometimes this is OK, but oftentimes we will want to assign these labels ourselves.
# 
# The list of row labels used in a DataFrame is known as an **Index**. We can assign values to it by using an `index` parameter in our constructor:

# CELL ********************

pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 
              'Sue': ['Pretty good.', 'Bland.']},
             index=['Product A', 'Product B'])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### Series
# 
# A Series, by contrast, is a sequence of data values. If a DataFrame is a table, a Series is a list. And in fact you can create one with nothing more than a list:

# CELL ********************

pd.Series([1, 2, 3, 4, 5]) 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# A Series is, in essence, a single column of a DataFrame. So you can assign row labels to the Series the same way as before, using an `index` parameter. However, a Series does not have a column name, it only has one overall `name`:

# CELL ********************

pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# The Series and the DataFrame are intimately related. It's helpful to think of a DataFrame as actually being just a bunch of Series "glued together". We'll see more of this in the next section of this tutorial.

# MARKDOWN ********************

# # Reading data files
# 
# Being able to create a DataFrame or Series by hand is handy. But, most of the time, we won't actually be creating our own data by hand. Instead, we'll be working with data that already exists.
# 
# Data can be stored in any of a number of different forms and formats. By far the most basic of these is the humble CSV file. When you open a CSV file you get something that looks like this:
# 
# ```
# Product A,Product B,Product C,
# 30,21,9,
# 35,34,1,
# 41,11,11
# ```
# 
# So a CSV file is a table of values separated by commas. Hence the name: "Comma-Separated Values", or CSV.
# 
# Let's now set aside our toy datasets and see what a real dataset looks like when we read it into a DataFrame. We'll use the `pd.read_csv()` function to read the data into a DataFrame. We will use the Fabric Lakehouse as a source for our files, /lakehouse/default/ points to the default lakehouse chosen in the notebook so make sure that you select the correct lakehouse !
# This goes thusly:

# CELL ********************

wine_reviews = pd.read_csv("/lakehouse/default/" + "Files/DMML_Aula0/wine-reviews/winemag-data_first150k.csv")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# We can use the `shape` attribute to check how large the resulting DataFrame is:

# CELL ********************

wine_reviews.shape

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# So our new DataFrame has 130,000 records split across 14 different columns. That's almost 2 million entries!
# 
# We can examine the contents of the resultant DataFrame using the `head()` command, which grabs the first five rows:

# CELL ********************

wine_reviews.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# The `pd.read_csv()` function is well-endowed, with over 30 optional parameters you can specify. For example, you can see in this dataset that the CSV file has a built-in index, which pandas did not pick up on automatically. To make pandas use that column for the index (instead of creating a new one from scratch), we can specify an `index_col`.

# CELL ********************


wine_reviews = pd.read_csv("/lakehouse/default/" + "Files/DMML_Aula0/wine-reviews/winemag-data_first150k.csv", index_col=0)
wine_reviews.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Your turn
# 
# If you haven't started the exercise, you can start with the exercise 1 notebook 
