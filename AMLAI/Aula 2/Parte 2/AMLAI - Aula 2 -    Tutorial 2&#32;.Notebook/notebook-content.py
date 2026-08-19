# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# # Advanced Recommendation Systems: Collaborative Filtering #
# 
# In the previous lesson, we learned about **Item-to-Item** and **User-to-User** collaborative filtering using similarity metrics. While these methods work well for smaller datasets, they face challenges with:
# 
# - **Scalability**: Computing similarities for millions of users and items is expensive
# - **Sparsity**: Most users only rate a tiny fraction of available items
# - **Cold Start**: Cannot recommend for new users or items
# 
# In this lesson, we'll explore **Matrix Factorization** techniques that address these challenges by learning latent features that explain user preferences and item characteristics.
# 
# # The Matrix Factorization Approach #
# 
# The key insight: We can approximate the entire user-item rating matrix $R$ as the product of two smaller matrices:
# 
# $$R \\approx P \\times Q^T$$
# 
# Where:
# - $R$: $m \\times n$ user-item rating matrix ($m$ users, $n$ items)
# - $P$: $m \\times k$ user feature matrix
# - $Q$: $n \\times k$ item feature matrix
# - $k$: number of latent features (much smaller than $m$ and $n$)
# 
# **Example:** For a movie recommendation system with $k=3$ latent features:
# - Feature 1 might represent "action vs. romance"
# - Feature 2 might represent "serious vs. comedic"
# - Feature 3 might represent "old vs. new"
# 
# <figure style="padding: 1em;">
# <img src=\"https://miro.medium.com/max/1400/1*9M5dPH9vFWnH5TsT6XkXTQ.png\" width=\"500\" alt=\"Matrix factorization visualization\">
# <figcaption style=\"textalign: center; font-style: italic\"><center>Matrix Factorization decomposes the rating matrix into user and item feature matrices.
# </center></figcaption>
# </figure>
# 
# # Prediction Formula #
# 
# To predict user $u$'s rating for item $i$:
# 
# $$\\hat{r}_{ui} = \\mu + b_u + b_i + p_u \\cdot q_i$$
# 
# Where:
# - $\\mu$: global average rating
# - $b_u$: user bias (some users rate higher/lower than average)
# - $b_i$: item bias (some items are rated higher/lower than average)
# - $p_u \\cdot q_i$: dot product of user and item latent feature vectors
# 
# This formula captures both global patterns (through biases) and personalized preferences (through latent features).


# MARKDOWN ********************

# # Stacking Dense Layers #
# 
# Now that we have some nonlinearity, let's see how we can stack layers to get complex data transformations.
# 
# <figure style="padding: 1em;">
# <img src="https://storage.googleapis.com/kaggle-media/learn/images/Y5iwFQZ.png" width="450" alt="An input layer, two hidden layers, and a final linear layer.">
# <figcaption style="textalign: center; font-style: italic"><center>A stack of dense layers makes a "fully-connected" network.
# </center></figcaption>
# </figure>
# 
# The layers before the output layer are sometimes called **hidden** since we never see their outputs directly.
# 
# Now, notice that the final (output) layer is a linear unit (meaning, no activation function). That makes this network appropriate to a regression task, where we are trying to predict some arbitrary numeric value. Other tasks (like classification) might require an activation function on the output.
# 
# ## Building Sequential Models ##
# 
# The `Sequential` model we've been using will connect together a list of layers in order from first to last: the first layer gets the input, the last layer produces the output. This creates the model in the figure above:


# CELL ********************

%pip install -q tensorflow

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    keras.Input(shape=[2]),
    # the hidden ReLU layers
    layers.Dense(units=4, activation='relu'),
    layers.Dense(units=3, activation='relu'),
    # the linear output layer 
    layers.Dense(units=1),
])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Be sure to pass all the layers together in a list, like `[layer, layer, layer, ...]`, instead of as separate arguments. To add an activation function to a layer, just give its name in the `activation` argument.
# 
# # Your Turn #
# 
# Now, [**create a deep neural network**] for the *Concrete* dataset.
