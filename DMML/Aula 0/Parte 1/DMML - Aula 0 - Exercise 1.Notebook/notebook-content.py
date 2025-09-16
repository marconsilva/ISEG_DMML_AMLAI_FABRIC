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
# The first step in most data analytics projects is reading the data file. In this exercise, you'll create Series and DataFrame objects, both by hand and by reading data files.
# 
# Run the code cell below to load libraries you will need (including code to check your answers).

# CELL ********************

%pip install /lakehouse/default/Files/Env/learntools-0.3.4-py2.py3-none-any.whl

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

import pandas as pd
pd.set_option('display.max_rows', 5)
from learntools.core import binder; binder.bind(globals())
from learntools.pandas.creating_reading_and_writing import *
print("Setup complete.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Exercises

# MARKDOWN ********************

# ## 1.
# 
# In the cell below, create a DataFrame `fruits` that looks like this:
# 
# ![](https://storage.googleapis.com/kaggle-media/learn/images/Ax3pp2A.png)

# CELL ********************

# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.
fruits = ____


# Check your answer
q1.check()
fruits

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

#q1.hint()
#q1.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 2.
# 
# Create a dataframe `fruit_sales` that matches the diagram below:
# 
# ![](https://storage.googleapis.com/kaggle-media/learn/images/CHPn7ZF.png)

# CELL ********************

# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.
fruit_sales = ____

# Check your answer
q2.check()
fruit_sales

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

#q2.hint()
#q2.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 3.
# 
# Create a variable `ingredients` with a Series that looks like:
# 
# ```
# Flour     4 cups
# Milk       1 cup
# Eggs     2 large
# Spam       1 can
# Name: Dinner, dtype: object
# ```

# CELL ********************

ingredients = ____

# Check your answer
q3.check()
ingredients

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

#q3.hint()
#q3.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 4.
# 
# Read the following csv dataset of wine reviews into a DataFrame called `reviews`:
# 
# ![](https://storage.googleapis.com/kaggle-media/learn/images/74RCZtU.png)
# 
# The filepath to the csv file is "/lakehouse/default/" + "Files/wine-reviews/winemag-data_first150k". The first few lines look like:
# 
# ```
# ,country,description,designation,points,price,province,region_1,region_2,variety,winery
# 0,US,"This tremendous 100% varietal wine[...]",Martha's Vineyard,96,235.0,California,Napa Valley,Napa,Cabernet Sauvignon,Heitz
# 1,Spain,"Ripe aromas of fig, blackberry and[...]",Carodorum Selección Especial Reserva,96,110.0,Northern Spain,Toro,,Tinta de Toro,Bodega Carmen Rodríguez
# ```

# CELL ********************

reviews = ____

# Check your answer
q4.check()
reviews

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

#q4.hint()
#q4.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 5.
# 
# Run the cell below to create and display a DataFrame called `animals`:

# CELL ********************

animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# In the cell below, write code to save this DataFrame in Lakehouse as a csv file with the name `cows_and_goats.csv`. Use the path "/lakehouse/default/Files/" to save the data.

# CELL ********************

# Your code goes here

# Check your answer
q5.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

#q5.hint()
#q5.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Keep going
# 
# Move on to learn about **[indexing, selecting and assigning](https://www.kaggle.com/residentmario/indexing-selecting-assigning)**.
