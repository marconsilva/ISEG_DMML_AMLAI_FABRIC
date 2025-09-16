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
# Maps allow us to transform data in a DataFrame or Series one value at a time for an entire column. However, often we want to group our data, and then do something specific to the group the data is in. 
# 
# As you'll learn, we do this with the `groupby()` operation.  We'll also cover some additional topics, such as more complex ways to index your DataFrames, along with how to sort your data.
# 
# **To start the exercise for this topic, please click [here](#$NEXT_NOTEBOOK_URL$).**
# 
# # Groupwise analysis
# 
# One function we've been using heavily thus far is the `value_counts()` function. We can replicate what `value_counts()` does by doing the following:

# CELL ********************

import pandas as pd
reviews = pd.read_csv("/lakehouse/default/" + "Files/DMML_Aula0/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

reviews.groupby('points').points.count()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# `groupby()` created a group of reviews which allotted the same point values to the given wines. Then, for each of these groups, we grabbed the `points()` column and counted how many times it appeared.  `value_counts()` is just a shortcut to this `groupby()` operation. 
# 
# We can use any of the summary functions we've used before with this data. For example, to get the cheapest wine in each point value category, we can do the following:

# CELL ********************

reviews.groupby('points').price.min()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# You can think of each group we generate as being a slice of our DataFrame containing only data with values that match. This DataFrame is accessible to us directly using the `apply()` method, and we can then manipulate the data in any way we see fit. For example, here's one way of selecting the name of the first wine reviewed from each winery in the dataset:

# CELL ********************

reviews.groupby('winery').apply(lambda df: df.title.iloc[0])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# For even more fine-grained control, you can also group by more than one column. For an example, here's how we would pick out the best wine by country _and_ province:

# CELL ********************

reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Another `groupby()` method worth mentioning is `agg()`, which lets you run a bunch of different functions on your DataFrame simultaneously. For example, we can generate a simple statistical summary of the dataset as follows:

# CELL ********************

reviews.groupby(['country']).price.agg([len, min, max])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Effective use of `groupby()` will allow you to do lots of really powerful things with your dataset.

# MARKDOWN ********************

# # Multi-indexes
# 
# In all of the examples we've seen thus far we've been working with DataFrame or Series objects with a single-label index. `groupby()` is slightly different in the fact that, depending on the operation we run, it will sometimes result in what is called a multi-index.
# 
# A multi-index differs from a regular index in that it has multiple levels. For example:

# CELL ********************

countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
countries_reviewed

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

mi = countries_reviewed.index
type(mi)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Multi-indices have several methods for dealing with their tiered structure which are absent for single-level indices. They also require two levels of labels to retrieve a value. Dealing with multi-index output is a common "gotcha" for users new to pandas.
# 
# The use cases for a multi-index are detailed alongside instructions on using them in the [MultiIndex / Advanced Selection](https://pandas.pydata.org/pandas-docs/stable/advanced.html) section of the pandas documentation.
# 
# However, in general the multi-index method you will use most often is the one for converting back to a regular index, the `reset_index()` method:

# CELL ********************

countries_reviewed.reset_index()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Sorting
# 
# Looking again at `countries_reviewed` we can see that grouping returns data in index order, not in value order. That is to say, when outputting the result of a `groupby`, the order of the rows is dependent on the values in the index, not in the data.
# 
# To get data in the order want it in we can sort it ourselves.  The `sort_values()` method is handy for this.

# CELL ********************

countries_reviewed = countries_reviewed.reset_index()
countries_reviewed.sort_values(by='len')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# `sort_values()` defaults to an ascending sort, where the lowest values go first. However, most of the time we want a descending sort, where the higher numbers go first. That goes thusly:

# CELL ********************

countries_reviewed.sort_values(by='len', ascending=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# To sort by index values, use the companion method `sort_index()`. This method has the same arguments and default order:

# CELL ********************

countries_reviewed.sort_index()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Finally, know that you can sort by more than one column at a time:

# CELL ********************

countries_reviewed.sort_values(by=['country', 'len'])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Your turn
# 
# If you haven't started the exercise, you can start the pratical exercise with the exercise notebook.
