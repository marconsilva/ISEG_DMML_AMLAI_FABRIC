# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
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

# Now that you can create your own line charts, it's time to learn about more chart types!  
# 
# > By the way, if this is your first experience with writing code in Python, you should be _very proud_ of all that you have accomplished so far, because it's never easy to learn a completely new skill!  If you stick with the course, you'll notice that everything will only get easier (while the charts you'll build will get more impressive!), since the code is pretty similar for all of the charts.  Like any skill, coding becomes natural over time, and with repetition.
# 
# In this tutorial, you'll learn about **bar charts** and **heatmaps**.
# 
# # Set up the notebook
# 
# As always, we begin by setting up the coding environment.  (_This code is hidden, but you can un-hide it by clicking on the "Code" button immediately below this text, on the right._)

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
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Select a dataset
# 
# In this tutorial, we'll work with a dataset from the US Department of Transportation that tracks flight delays.
# 
# Opening this CSV file in Excel shows a row for each month (where `1` = January, `2` = February, etc) and a column for each airline code.
# 
# ![tut2_flight_delay_head](https://storage.googleapis.com/kaggle-media/learn/images/5nYs9se.png)
# 
# Each entry shows the average arrival delay (in minutes) for a different airline and month (all in year 2015).  Negative entries denote flights that (_on average_) tended to arrive early.  For instance, the average American Airlines flight (_airline code: **AA**_) in January arrived roughly 7 minutes late, and the average Alaska Airlines flight (_airline code: **AS**_) in April arrived roughly 3 minutes early.
# 
# # Load the data
# 
# As before, we load the dataset using the `pd.read_csv` command.

# CELL ********************

# Path of the file to read
flight_filepath = "/lakehouse/default/Files/DMML_Aula2/flight_delays.csv"

# Read the file into a variable flight_data
flight_data = pd.read_csv(flight_filepath, index_col="Month")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# You may notice that the code is slightly shorter than what we used in the previous tutorial.  In this case, since the row labels (from the `'Month'` column) don't correspond to dates, we don't add `parse_dates=True` in the parentheses.  But, we keep the first two pieces of text as before, to provide both: 
# - the filepath for the dataset (in this case, `flight_filepath`), and 
# - the name of the column that will be used to index the rows (in this case, `index_col="Month"`). 
# 
# # Examine the data
# 
# Since the dataset is small, we can easily print all of its contents.  This is done by writing a single line of code with just the name of the dataset.

# CELL ********************

# Print the data
flight_data

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Bar chart
# 
# Say we'd like to create a bar chart showing the average arrival delay for Spirit Airlines (_airline code: **NK**_) flights, by month.  

# CELL ********************

# Set the width and height of the figure
plt.figure(figsize=(10,6))

# Add title
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")

# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=flight_data.index, y=flight_data['NK'])

# Add label for vertical axis
plt.ylabel("Arrival delay (in minutes)")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# The commands for customizing the text (title and vertical axis label) and size of the figure are familiar from the previous tutorial.  The code that creates the bar chart is new:
# 
# ```python
# # Bar chart showing average arrival delay for Spirit Airlines flights by month
# sns.barplot(x=flight_data.index, y=flight_data['NK'])
# ```
# It has three main components:
# - `sns.barplot` - This tells the notebook that we want to create a bar chart.
#  - _Remember that `sns` refers to the [seaborn](https://seaborn.pydata.org/) package, and all of the commands that you use to create charts in this course will start with this prefix._
# - `x=flight_data.index` - This determines what to use on the horizontal axis.  In this case, we have selected the column that **_index_**es the rows (in this case, the column containing the months).
# - `y=flight_data['NK']` - This sets the column in the data that will be used to determine the height of each bar.  In this case, we select the `'NK'` column.
# 
# > **Important Note**: You must select the indexing column with `flight_data.index`, and it is not possible to use `flight_data['Month']` (_which will return an error_).  This is because when we loaded the dataset, the `"Month"` column was used to index the rows.  **We always have to use this special notation to select the indexing column.**
# 
# # Heatmap
# 
# We have one more plot type to learn about: **heatmaps**! 
# 
# In the code cell below, we create a heatmap to quickly visualize patterns in `flight_data`.  Each cell is color-coded according to its corresponding value.


# CELL ********************

# Set the width and height of the figure
plt.figure(figsize=(14,7))

# Add title
plt.title("Average Arrival Delay for Each Airline, by Month")

# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=flight_data, annot=True)

# Add label for horizontal axis
plt.xlabel("Airline")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# The relevant code to create the heatmap is as follows:
# ```python
# # Heatmap showing average arrival delay for each airline by month
# sns.heatmap(data=flight_data, annot=True)
# ```
# This code has three main components:
# - `sns.heatmap` - This tells the notebook that we want to create a heatmap.
# - `data=flight_data` - This tells the notebook to use all of the entries in `flight_data` to create the heatmap.
# - `annot=True` - This ensures that the values for each cell appear on the chart.  (_Leaving this out removes the numbers from each of the cells!_)
# 
# _What patterns can you detect in the table?  For instance, if you look closely, the months toward the end of the year (especially months 9-11) appear relatively dark for all airlines.  This suggests that airlines are better (on average) at keeping schedule during these months!_  
# 
# # What's next?
# 
# Create your own visualizations with a **[coding exercise]** in the next notebook!
