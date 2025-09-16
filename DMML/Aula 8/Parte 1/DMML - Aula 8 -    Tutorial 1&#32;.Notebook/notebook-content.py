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

# # Introduction #
# 
# This lesson and the next make use of what are known as *unsupervised learning* algorithms. Unsupervised algorithms don't make use of a target; instead, their purpose is to learn some property of the data, to represent the structure of the features in a certain way. In the context of feature engineering for prediction, you could think of an unsupervised algorithm as a "feature discovery" technique.
# 
# **Clustering** simply means the assigning of data points to groups based upon how similar the points are to each other. A clustering algorithm makes "birds of a feather flock together," so to speak.
# 
# When used for feature engineering, we could attempt to discover groups of customers representing a market segment, for instance, or geographic areas that share similar weather patterns. Adding a feature of cluster labels can help machine learning models untangle complicated relationships of space or proximity.
# 
# # Cluster Labels as a Feature #
# 
# Applied to a single real-valued feature, clustering acts like a traditional "binning" or ["discretization"](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_discretization_classification.html) transform. On multiple features, it's like "multi-dimensional binning" (sometimes called *vector quantization*).
# 
# <figure style="padding: 1em;">
# <img src="https://storage.googleapis.com/kaggle-media/learn/images/sr3pdYI.png" width=800, alt="">
# <figcaption style="textalign: center; font-style: italic"><center><strong>Left:</strong> Clustering a single feature. <strong>Right:</strong> Clustering across two features.
# </center></figcaption>
# </figure>
# 
# Added to a dataframe, a feature of cluster labels might look like this:
# 
# | Longitude | Latitude | Cluster |
# |-----------|----------|---------|
# | -93.619   | 42.054   | 3       |
# | -93.619   | 42.053   | 3       |
# | -93.638   | 42.060   | 1       |
# | -93.602   | 41.988   | 0       |


# MARKDOWN ********************

# It's important to remember that this `Cluster` feature is categorical. Here, it's shown with a label encoding (that is, as a sequence of integers) as a typical clustering algorithm would produce; depending on your model, a one-hot encoding may be more appropriate.
# 
# The motivating idea for adding cluster labels is that the clusters will break up complicated relationships across features into simpler chunks. Our model can then just learn the simpler chunks one-by-one instead having to learn the complicated whole all at once. It's a "divide and conquer" strategy.
# 
# <figure style="padding: 1em;">
# <img src="https://storage.googleapis.com/kaggle-media/learn/images/rraXFed.png" width=800, alt="">
# <figcaption style="textalign: center; font-style: italic"><center>Clustering the YearBuilt feature helps this linear model learn its relationship to SalePrice.
# </center></figcaption>
# </figure>
# 
# The figure shows how clustering can improve a simple linear model. The curved relationship between the `YearBuilt` and `SalePrice` is too complicated for this kind of model -- it *underfits*. On smaller chunks however the relationship is *almost* linear, and that the model can learn easily.
# 
# # k-Means Clustering #
# 
# There are a great many clustering algorithms. They differ primarily in how they measure "similarity" or "proximity" and in what kinds of features they work with. The algorithm we'll use, k-means, is intuitive and easy to apply in a feature engineering context. Depending on your application another algorithm might be more appropriate.
# 
# **K-means clustering** measures similarity using ordinary straight-line distance (Euclidean distance, in other words). It creates clusters by placing a number of points, called **centroids**, inside the feature-space. Each point in the dataset is assigned to the cluster of whichever centroid it's closest to. The "k" in "k-means" is how many centroids (that is, clusters) it creates. You define the k yourself.
# 
# You could imagine each centroid capturing points through a sequence of radiating circles. When sets of circles from competing centroids overlap they form a line. The result is what's called a **Voronoi tessallation**. The tessallation shows you to what clusters future data will be assigned; the tessallation is essentially what k-means learns from its training data.
# 
# The clustering on the [*Ames*] dataset above is a k-means clustering. Here is the same figure with the tessallation and centroids shown.
# 
# <figure style="padding: 1em;">
# <img src="https://storage.googleapis.com/kaggle-media/learn/images/KSoLd3o.jpg" width=450, alt="">
# <figcaption style="textalign: center; font-style: italic"><center>K-means clustering creates a Voronoi tessallation of the feature space.
# </center></figcaption>
# </figure>
# 
# Let's review how the k-means algorithm learns the clusters and what that means for feature engineering. We'll focus on three parameters from scikit-learn's implementation: `n_clusters`, `max_iter`, and `n_init`.
# 
# It's a simple two-step process. The algorithm starts by randomly initializing some predefined number (`n_clusters`) of centroids. It then iterates over these two operations:
# 1. assign points to the nearest cluster centroid
# 2. move each centroid to minimize the distance to its points
# 
# It iterates over these two steps until the centroids aren't moving anymore, or until some maximum number of iterations has passed (`max_iter`).
# 
# It often happens that the initial random position of the centroids ends in a poor clustering. For this reason the algorithm repeats a number of times (`n_init`) and returns the clustering that has the least total distance between each point and its centroid, the optimal clustering.
# 
# The animation below shows the algorithm in action. It illustrates the dependence of the result on the initial centroids and the importance of iterating until convergence.
# 
# <figure style="padding: 1em;">
# <img src="https://storage.googleapis.com/kaggle-media/learn/images/tBkCqXJ.gif" width=550, alt="">
# <figcaption style="textalign: center; font-style: italic"><center>The K-means clustering algorithm on Airbnb rentals in NYC.
# </center></figcaption>
# </figure>
# 
# You may need to increase the `max_iter` for a large number of clusters or `n_init` for a complex dataset. Ordinarily though the only parameter you'll need to choose yourself is `n_clusters` (k, that is). The best partitioning for a set of features depends on the model you're using and what you're trying to predict, so it's best to tune it like any hyperparameter (through cross-validation, say).
# 
# # Example - California Housing #
# 
# As spatial features, [*California Housing*]'s `'Latitude'` and `'Longitude'` make natural candidates for k-means clustering. In this example we'll cluster these with `'MedInc'` (median income) to create economic segments in different regions of California.


# CELL ********************

#$HIDE_INPUT$
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

import mlflow
mlflow.autolog(disable=True)


plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

df = pd.read_csv("/lakehouse/default/Files/DMML_Aula8/housing.csv")
X = df.loc[:, ["MedInc", "Latitude", "Longitude"]]
X.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Since k-means clustering is sensitive to scale, it can be a good idea rescale or normalize data with extreme values. Our features are already roughly on the same scale, so we'll leave them as-is.

# CELL ********************

# Create cluster feature
kmeans = KMeans(n_clusters=6)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")

X.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Now let's look at a couple plots to see how effective this was. First, a scatter plot that shows the geographic distribution of the clusters. It seems like the algorithm has created separate segments for higher-income areas on the coasts.

# CELL ********************

sns.relplot(
    x="Longitude", y="Latitude", hue="Cluster", data=X, height=6,
);

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# The target in this dataset is `MedHouseVal` (median house value). These box-plots show the distribution of the target within each cluster. If the clustering is informative, these distributions should, for the most part, separate across `MedHouseVal`, which is indeed what we see.

# CELL ********************

X["MedHouseVal"] = df["MedHouseVal"]
sns.catplot(x="MedHouseVal", y="Cluster", data=X, kind="boxen", height=6);

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Choosing the value of k - Elbow method #
# 
# The K-Means algorithm depends upon finding the number of clusters and data labels for a pre-defined value of K. To find the number of clusters in the data, we need to run the K-Means clustering algorithm for different values of K and compare the results. So, the performance of K-Means algorithm depends upon the value of K. We should choose the optimal value of K that gives us best performance. There are different techniques available to find the optimal value of K. The most common technique is the elbow method which is described below.
# 
# The elbow method is used to determine the optimal number of clusters in K-means clustering. The elbow method plots the value of the cost function produced by different values of K. The below diagram shows how the elbow method works:-


# MARKDOWN ********************

# ![Untitled.png](attachment:034e3300-b3a3-48f9-a794-d6a36272d6aa.png)

# ATTACHMENTS ********************

# ATTA {
# ATTA   "034e3300-b3a3-48f9-a794-d6a36272d6aa.png": {
# ATTA     "image/png": "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EdIYbhfV+lWrVk3q4nZRW9E6HgXZ4S1btkwqV64sgwYNMj+HqjjWD40hjhs6WQWpuPcfhnEeeOABOffcc+Xss8+WW265RR588MFg6EeyflEvl4BxJxivy8nJCTo7O9s4KyvLnFbBeE8sh943enTwnj17jNEaw7t27ZKdO3fKwYMHzekNnJSUZOy8r/OeOIXE6xH0U6dOlVKlSkmnTp1k06ZNpqcI48DDOAXW3st5P3jHjh3mtatWrTI9cnjFihXGy5cvN8aHHF67dq15H+29cIYBo+dz1113ybhx42ThwoXGCxYskPnz5wc9b948+fHHH+Wnn37K9x5Lly41Rhg6Rq8cr501a5bMnDnTeMaMGTJ9+vQTPGfOHMN2+PAPP/wgJUqUMPsJf4e//fZbmTJliumhwfiwffHFF8aff/658eTJk+Wbb76R7777zhjLwNOmTTP7HcbfYbz2008/NY0ujG2HP/nkExk7dqx8/PHHxngNXtu3b1+56aabpGXLljJx4kTTq4YnTJhgXjNmzBgZMWJE0GhAHQ8fPtx42LBhpleH13700UdBo/eE02UYjQleM3ToUBk4cKDxgAEDjPv372/cr18/Y4TqP//5T3n99dflvffeMx4yZIjx4MGDTcDBYPTp00d69eolPXv2DPqtt94y7tGjh7z55pvGeM27774r77zzjmlMHL/99tvmbw6je/fu0rVr16C7dOlifN1115l1wucbxt+6detm1hEh9+qrrwbtXrZDhw6mt4szFbhdu3by/PPPB/3cc88Z4zUvvfSS8Ysvvmh+di/nLPvMM8+YDsP9998vV111lZQtW1Yef/xx49atWxu3atVKnnjiCdP5wmth/Ayj8wM7yzz66KPSokULefjhh6V58+bGzZo1k6ZNmwbdpEkT83vndeFee9ttt8kVV1whNWrUkPvuu08aNmwoDRo0MK5fv34+42/YBrwu3Gvr1asnderUkZo1axrfc889hg1Xr17dGI1w1apVze+c1zivc7/GeR0aSDRIFSpUkD//+c9y1llnyRlnnCG/+c1vzP///e9/m88OhslORu58jmrYY+fiVOQf//iH/Otf/zIr+p///EduuOEG4//+979B33jjjeZLHfo75/fwzTffbIxW7n//+5/cfvvtxnfccYfxnXfeaYwARWjB+Bl/A+eiiy6Sv/zlL+bv7teULFlS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"
# ATTA   }
# ATTA }

# MARKDOWN ********************

# You can see in this example that 3 appears to be the correct number of cluster

# MARKDOWN ********************

# # Your Turn #
# 
# [**Add a feature of cluster labels**] to *Ames* and learn about another kind of feature clustering can create.
