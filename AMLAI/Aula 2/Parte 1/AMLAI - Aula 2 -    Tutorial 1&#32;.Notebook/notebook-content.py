# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# # Introduction to Recommendation Systems #
# 
# In this lesson we're going to explore how recommendation systems work - the technology behind Netflix suggestions, Amazon product recommendations, and Spotify playlists. We'll learn how computers can predict what you might like based on patterns in user behavior.
# 
# The key idea is **similarity**: if you liked certain items in the past, you'll probably like similar items in the future. Or if you're similar to other users, you'll probably like what they like!
# 
# # What are Recommendation Systems? #
# 
# Recommendation systems are algorithms designed to suggest relevant items to users. They power:
# - **E-commerce**: "Customers who bought this also bought..."
# - **Streaming Services**: Netflix movie suggestions, Spotify playlists
# - **Social Media**: Friend suggestions, content feeds
# - **News**: Personalized article recommendations
# 
# There are two main approaches we'll explore today:
# 1. **Item-to-Item Recommendations**: Find items similar to what you liked
# 2. **User-to-User Recommendations**: Find users similar to you and recommend what they like
# 
# # The User-Item Matrix #
# 
# The foundation of recommendation systems is the **user-item interaction matrix**. Each row represents a user, each column represents an item, and the values represent interactions (ratings, purchases, views).
# 
# Example movie rating matrix:
# ```
#            Movie1  Movie2  Movie3  Movie4
# User1        5       3       ?       1
# User2        4       ?       5       ?
# User3        ?       2       4       5
# User4        3       4       ?       2
# ```
# 
# The `?` marks represent missing values - movies the user hasn't rated. **Our goal is to predict these missing values!**
# 
# # Similarity Metrics #
# 
# To find similar items or users, we need to measure similarity. The most common metrics are:
# 
# ## 1. Cosine Similarity
# 
# Measures the cosine of the angle between two vectors. Values range from -1 to 1.
# 
# $$\text{cosine}(A, B) = \frac{A \cdot B}{||A|| \times ||B||} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}$$
# 
# ## 2. Pearson Correlation
# 
# Measures linear correlation between two variables. Values range from -1 to 1.
# 
# $$\text{pearson}(A, B) = \frac{\sum_{i=1}^{n}(A_i - \bar{A})(B_i - \bar{B})}{\sqrt{\sum_{i=1}^{n}(A_i - \bar{A})^2} \times \sqrt{\sum_{i=1}^{n}(B_i - \bar{B})^2}}$$
# 
# ## 3. Euclidean Distance
# 
# Measures straight-line distance between two points. Smaller = more similar.
# 
# $$\text{euclidean}(A, B) = \sqrt{\sum_{i=1}^{n}(A_i - B_i)^2}$$
# 
# <blockquote style="margin-right:auto; margin-left:auto; background-color: #ebf9ff; padding: 1em; margin:24px;">
#     <strong>Choosing the Right Metric</strong><br>
# <strong>Cosine similarity</strong> is best when magnitude doesn't matter (e.g., some users rate everything high, others rate everything low). <strong>Pearson correlation</strong> is good when you want to account for user bias in ratings. <strong>Euclidean distance</strong> is simple but sensitive to scale.
# </blockquote>

# # Item-to-Item Collaborative Filtering #
# 
# Item-to-Item recommendation finds items similar to ones you've already interacted with.
# 
# **Algorithm Steps:**
# 1. For each pair of items, compute similarity using user ratings
# 2. For a target user, identify items they've rated highly
# 3. Find similar items to those highly-rated items
# 4. Recommend the most similar items they haven't seen
# 
# **Mathematical Formula:**
# 
# To predict user $u$'s rating for item $i$:
# 
# $$\hat{r}_{ui} = \frac{\sum_{j \in N(i)} sim(i, j) \times r_{uj}}{\sum_{j \in N(i)} |sim(i, j)|}$$
# 
# Where:
# - $N(i)$ = set of items similar to item $i$ that user $u$ has rated
# - $sim(i, j)$ = similarity between items $i$ and $j$
# - $r_{uj}$ = user $u$'s rating of item $j$
# 
# **Advantages:**
# - More stable than user-to-user (item relationships don't change as quickly)
# - Scalable for many users
# - Easy to explain: "You liked X, so you might like Y"
# 
# **Disadvantages:**
# - Can't handle new items (cold start problem)
# - Limited by item features captured in ratings

# MARKDOWN ********************

# # User-to-User Collaborative Filtering #
# 
# User-to-User recommendation finds users similar to you and recommends what they liked.
# 
# **Algorithm Steps:**
# 1. For each pair of users, compute similarity based on their rating patterns
# 2. For a target user, find the most similar users (neighbors)
# 3. Identify items that similar users liked but target user hasn't seen
# 4. Recommend those items, weighted by user similarity
# 
# **Mathematical Formula:**
# 
# To predict user $u$'s rating for item $i$:
# 
# $$\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N(u)} sim(u, v) \times (r_{vi} - \bar{r}_v)}{\sum_{v \in N(u)} |sim(u, v)|}$$
# 
# Where:
# - $N(u)$ = set of users similar to user $u$
# - $sim(u, v)$ = similarity between users $u$ and $v$
# - $r_{vi}$ = user $v$'s rating of item $i$
# - $\bar{r}_u$, $\bar{r}_v$ = average ratings for users $u$ and $v$
# 
# **Advantages:**
# - Can discover unexpected recommendations
# - Works well with sufficient user data
# - Captures complex taste patterns
# 
# **Disadvantages:**
# - Scalability issues with many users
# - User preferences change over time
# - Cold start problem for new users

# MARKDOWN ********************

# # Implementation Example #
# 
# Let's implement a simple item-to-item recommender system using Python:

# CELL ********************

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Create a sample user-item rating matrix
ratings_data = {
    'User1': [5, 3, 0, 1, 4],
    'User2': [4, 0, 5, 0, 3],
    'User3': [0, 2, 4, 5, 0],
    'User4': [3, 4, 0, 2, 5]
}

items = ['Movie_A', 'Movie_B', 'Movie_C', 'Movie_D', 'Movie_E']
ratings_df = pd.DataFrame(ratings_data, index=items)

print("User-Item Rating Matrix:")
print(ratings_df)
print("\nNote: 0 means the user hasn't rated the item yet")

# Calculate item-to-item similarity matrix
item_similarity = cosine_similarity(ratings_df)
item_similarity_df = pd.DataFrame(item_similarity, 
                                  index=items, 
                                  columns=items)

print("\nItem-to-Item Similarity Matrix:")
print(item_similarity_df.round(3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Making Recommendations ##
# 
# Now let's use this similarity matrix to make recommendations:

# CELL ********************

def get_item_recommendations(user, item, ratings_df, similarity_df, k=3):
    """
    Get recommendations based on item-to-item similarity.
    
    Parameters:
    - user: target user
    - item: item the user liked
    - ratings_df: user-item rating matrix
    - similarity_df: item-to-item similarity matrix
    - k: number of recommendations
    """
    # Get items similar to the given item
    similar_items = similarity_df[item].sort_values(ascending=False)[1:k+1]
    
    # Filter out items the user has already rated
    user_unrated = ratings_df[user][ratings_df[user] == 0].index
    recommendations = similar_items[similar_items.index.isin(user_unrated)]
    
    return recommendations

# Example: User2 liked Movie_C, what else might they like?
recommendations = get_item_recommendations('User2', 'Movie_C', 
                                           ratings_df, item_similarity_df)

print("User2 liked Movie_C. Based on item similarity, we recommend:")
for item, similarity in recommendations.items():
    print(f"  {item}: similarity score = {similarity:.3f}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Key Concepts Summary #
# 
# - **User-Item Matrix**: Core data structure with users as rows, items as columns
# - **Similarity Metrics**: Cosine, Pearson, Euclidean for measuring similarity
# - **Item-to-Item**: Recommend items similar to what user liked
# - **User-to-User**: Recommend items that similar users liked
# - **Cold Start**: Challenge of recommending for new users/items
# - **Sparsity**: Most users only rate a small fraction of items
# 
# # Your Turn #
# 
# Now, move to the [**Exercise notebook**] to implement your own recommendation system with real data!
