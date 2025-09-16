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

# # Introduction
# 
# Oftentimes data will come to us with column names, index names, or other naming conventions that we are not satisfied with. In that case, you'll learn how to use pandas functions to change the names of the offending entries to something better.
# 
# You'll also explore how to combine data from multiple DataFrames and/or Series.
# 
# **To start the exercise for this topic, go to the notebook.**
# 
# # Renaming
# 
# The first function we'll introduce here is `rename()`, which lets you change index names and/or column names. For example, to change the `points` column in our dataset to `score`, we would do:

# CELL ********************

import pandas as pd
pd.set_option('display.max_rows', 5)
reviews = pd.read_csv("/lakehouse/default/Files/DMML_Aula0/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

reviews.rename(columns={'points': 'score'})

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# `rename()` lets you rename index _or_ column values by specifying a `index` or `column` keyword parameter, respectively. It supports a variety of input formats, but usually a Python dictionary is the most convenient. Here is an example using it to rename some elements of the index.

# CELL ********************

reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# You'll probably rename columns very often, but rename index values very rarely.  For that, `set_index()` is usually more convenient.
# 
# Both the row index and the column index can have their own `name` attribute. The complimentary `rename_axis()` method may be used to change these names. For example:

# CELL ********************

reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Combining
# 
# When performing operations on a dataset, we will sometimes need to combine different DataFrames and/or Series in non-trivial ways. Pandas has three core methods for doing this. In order of increasing complexity, these are `concat()`, `join()`, and `merge()`. Most of what `merge()` can do can also be done more simply with `join()`, so we will omit it and focus on the first two functions here.
# 
# The simplest combining method is `concat()`. Given a list of elements, this function will smush those elements together along an axis.
# 
# This is useful when we have data in different DataFrame or Series objects but having the same fields (columns). One example: the [YouTube Videos dataset](https://www.kaggle.com/datasnaek/youtube-new), which splits the data up based on country of origin (e.g. Canada and the UK, in this example). If we want to study multiple countries simultaneously, we can use `concat()` to smush them together:

# CELL ********************

canadian_youtube = pd.read_csv("/lakehouse/default/Files/DMML_Aula0/youtube-new/CAvideos.csv")
british_youtube = pd.read_csv("/lakehouse/default/Files/DMML_Aula0/youtube-new/GBvideos.csv")

pd.concat([canadian_youtube, british_youtube])



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# The middlemost combiner in terms of complexity is `join()`. `join()` lets you combine different DataFrame objects which have an index in common. For example, to pull down videos that happened to be trending on the same day in _both_ Canada and the UK, we could do the following:

# CELL ********************

left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])

left.join(right, lsuffix='_CAN', rsuffix='_UK')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# The `lsuffix` and `rsuffix` parameters are necessary here because the data has the same column names in both British and Canadian datasets. If this wasn't true (because, say, we'd renamed them beforehand) we wouldn't need them.
# 
# # Your turn
# 
# If you haven't started the exercise, you can go to notebook and start
