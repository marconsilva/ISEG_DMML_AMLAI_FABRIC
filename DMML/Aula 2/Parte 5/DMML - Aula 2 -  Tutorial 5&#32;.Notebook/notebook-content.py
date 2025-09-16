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

# In this tutorial you'll learn all about **histograms** and **density plots**.
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
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Select a dataset
# 
# We'll work with a dataset of 150 different flowers, or 50 each from three different species of iris (*Iris setosa*, *Iris versicolor*, and *Iris virginica*).
# 
# ![tut4_iris](https://storage.googleapis.com/kaggle-media/learn/images/RcxYYBA.png)
# 
# # Load and examine the data
# 
# Each row in the dataset corresponds to a different flower.  There are four measurements: the sepal length and width, along with the petal length and width.  We also keep track of the corresponding species. 

# CELL ********************

# Path of the file to read
iris_filepath = "/lakehouse/default/Files/DMML_Aula2/iris.csv"

# Read the file into a variable iris_data
iris_data = pd.read_csv(iris_filepath, index_col="Id")

# Print the first 5 rows of the data
iris_data.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Histograms
# 
# Say we would like to create a **histogram** to see how petal length varies in iris flowers.  We can do this with the `sns.histplot` command.  

# CELL ********************

# Histogram 
sns.histplot(iris_data['Petal Length (cm)'])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# In the code cell above, we had to supply the command with the column we'd like to plot (_in this case, we chose `'Petal Length (cm)'`_).
# 
# # Density plots
# 
# The next type of plot is a **kernel density estimate (KDE)** plot.  In case you're not familiar with KDE plots, you can think of it as a smoothed histogram.   
# 
# To make a KDE plot, we use the `sns.kdeplot` command.  Setting `shade=True` colors the area below the curve (_and `data=` chooses the column we would like to plot_).

# CELL ********************

# KDE plot 
sns.kdeplot(data=iris_data['Petal Length (cm)'], shade=True)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # 2D KDE plots
# 
# We're not restricted to a single column when creating a KDE plot.  We can create a **two-dimensional (2D) KDE plot** with the `sns.jointplot` command.
# 
# In the plot below, the color-coding shows us how likely we are to see different combinations of sepal width and petal length, where darker parts of the figure are more likely. 

# CELL ********************

# 2D KDE plot
sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Note that in addition to the 2D KDE plot in the center,
# - the curve at the top of the figure is a KDE plot for the data on the x-axis (in this case, `iris_data['Petal Length (cm)']`), and
# - the curve on the right of the figure is a KDE plot for the data on the y-axis (in this case, `iris_data['Sepal Width (cm)']`).

# MARKDOWN ********************

# # Color-coded plots
# 
# For the next part of the tutorial, we'll create plots to understand differences between the species.  
# 
# We can create three different histograms (one for each species) of petal length by using the `sns.histplot` command (_as above_).  
# - `data=` provides the name of the variable that we used to read in the data
# - `x=` sets the name of column with the data we want to plot
# - `hue=` sets the column we'll use to split the data into different histograms 

# CELL ********************

# Histograms for each species
sns.histplot(data=iris_data, x='Petal Length (cm)', hue='Species')

# Add title
plt.title("Histogram of Petal Lengths, by Species")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# We can also create a KDE plot for each species by using `sns.kdeplot` (_as above_).  The functionality for `data`, `x`, and `hue` are identical to when we used `sns.histplot` above.  Additionally, we set `shade=True` to color the area below each curve.

# CELL ********************

# KDE plots for each species
sns.kdeplot(data=iris_data, x='Petal Length (cm)', hue='Species', shade=True)

# Add title
plt.title("Distribution of Petal Lengths, by Species")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# One interesting pattern that can be seen in plots is that the plants seem to belong to one of two groups, where _Iris versicolor_ and _Iris virginica_ seem to have similar values for petal length, while _Iris setosa_ belongs in a category all by itself. 
# 
# In fact, according to this dataset, we might even be able to classify any iris plant as *Iris setosa* (as opposed to *Iris versicolor* or *Iris virginica*) just by looking at the petal length: if the petal length of an iris flower is less than 2 cm, it's most likely to be *Iris setosa*!

# MARKDOWN ********************

# # What's next?
# 
# Put your new skills to work in a **[coding exercise]**!
