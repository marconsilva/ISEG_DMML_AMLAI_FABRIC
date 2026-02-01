# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "a5591839-f387-4a67-a52e-dac9b3ea21b0",
# META       "default_lakehouse_name": "DataScienceLearnLakehouse",
# META       "default_lakehouse_workspace_id": "03f3982f-785f-4a2f-8ec0-4be54060ee7b"
# META     },
# META     "environment": {
# META       "environmentId": "2b9c63f7-1498-40e2-81b9-a8ccb1b5f193",
# META       "workspaceId": "03f3982f-785f-4a2f-8ec0-4be54060ee7b"
# META     }
# META   }
# META }

# MARKDOWN ********************

# # Introduction #
# 
# In the tutorial, we learned about recommendation systems and how to use similarity metrics to suggest items to users. We explored both **Item-to-Item** and **User-to-User** collaborative filtering approaches.
# 
# In these exercises, you'll implement your own recommendation system using a movie rating dataset. You'll calculate similarities, make predictions, and evaluate your recommendations. Let's get started!

# CELL ********************

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
mlflow.autolog(disable=True)

plt.style.use('seaborn-v0_8-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

print("Libraries loaded successfully!")
print("Ready to build recommendation systems!")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# For these exercises, we'll create a sample movie rating dataset. In a real scenario, you would use a larger dataset like MovieLens.
# 
# Run the next code cell without changes to load the dataset.

# CELL ********************

# Create a sample movie ratings dataset
np.random.seed(42)

users = ['Alice', 'Bob', 'Carol', 'David', 'Eve', 'Frank']
movies = ['Inception', 'Titanic', 'Avatar', 'The Matrix', 'Interstellar', 
          'Pulp Fiction', 'Forrest Gump', 'The Godfather']

# Create a sparse rating matrix (many users haven't rated many movies)
ratings = np.array([
    [5, 3, 0, 4, 5, 0, 0, 4],  # Alice
    [4, 0, 5, 0, 4, 3, 0, 0],  # Bob
    [0, 4, 3, 0, 0, 5, 4, 5],  # Carol
    [3, 5, 0, 2, 3, 0, 4, 0],  # David
    [5, 0, 4, 5, 0, 0, 0, 3],  # Eve
    [0, 3, 4, 0, 0, 4, 5, 4],  # Frank
])

ratings_df = pd.DataFrame(ratings, index=users, columns=movies)
print("Movie Ratings Matrix (0 = not rated):")
print(ratings_df)
print(f"\nMatrix shape: {ratings_df.shape}")
print(f"Sparsity: {(ratings == 0).sum() / ratings.size * 100:.1f}% of ratings are missing")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # 1) Calculate Item-to-Item Similarity #
# 
# Your first task is to calculate the similarity between all pairs of movies. Use **cosine similarity** to measure how similar each movie's rating pattern is to every other movie.
# 
# Remember: We transpose the matrix so that movies become rows, and we can compare their rating vectors.
# 
# **Hint:** Use `cosine_similarity()` from sklearn on the transposed ratings matrix.

# CELL ********************

# YOUR CODE HERE
# Calculate item-to-item (movie-to-movie) similarity
item_similarity = ____
item_similarity_df = pd.DataFrame(item_similarity, 
                                  index=movies, 
                                  columns=movies)

print("Item-to-Item Similarity Matrix:")
print(item_similarity_df.round(3))

# Visualize the similarity matrix
plt.figure(figsize=(10, 8))
sns.heatmap(item_similarity_df, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Movie Similarity Heatmap (Cosine Similarity)')
plt.tight_layout()
plt.show()

# Expected output: A symmetric matrix with 1.0 on the diagonal

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# <details>
# <summary>Click here for Solution to Exercise 1</summary>
# 
# ```python
# item_similarity = cosine_similarity(ratings_df.T)
# item_similarity_df = pd.DataFrame(item_similarity, 
#                                   index=movies, 
#                                   columns=movies)
# ```
# </details>

# MARKDOWN ********************

# # 2) Make Item-Based Recommendations #
# 
# Now implement a function to recommend movies to a user based on **item-to-item similarity**.
# 
# The algorithm should:
# 1. Find movies the user has rated highly (rating >= 4)
# 2. For each highly-rated movie, find similar movies
# 3. Recommend similar movies that the user hasn't seen yet
# 4. Rank recommendations by similarity score

# CELL ********************

def recommend_items(user, ratings_df, similarity_df, n_recommendations=3):
    """
    Recommend items based on item-to-item similarity.
    
    Parameters:
    - user: name of the user
    - ratings_df: user-item ratings matrix
    - similarity_df: item-item similarity matrix
    - n_recommendations: number of items to recommend
    
    Returns:
    - List of recommended items with scores
    """
    # YOUR CODE HERE
    # Step 1: Get user's ratings
    user_ratings = ____
    
    # Step 2: Find items user rated highly (>= 4)
    liked_items = ____
    
    # Step 3: Find items user hasn't rated
    unrated_items = ____
    
    # Step 4: Calculate recommendation scores
    recommendations = {}
    for item in unrated_items:
        # Calculate weighted score based on similarity to liked items
        score = 0
        # YOUR CODE HERE
        ____
        recommendations[item] = score
    
    # Step 5: Sort and return top N
    sorted_recommendations = sorted(recommendations.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)
    return sorted_recommendations[:n_recommendations]

# Test your function
user_to_recommend = 'Bob'
recommendations = recommend_items(user_to_recommend, ratings_df, item_similarity_df)

print(f"\nTop recommendations for {user_to_recommend}:")
for item, score in recommendations:
    print(f"  {item}: score = {score:.3f}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# <details>
# <summary>Click here for Solution to Exercise 2</summary>
# 
# ```python
# def recommend_items(user, ratings_df, similarity_df, n_recommendations=3):
#     user_ratings = ratings_df.loc[user]
#     liked_items = user_ratings[user_ratings >= 4].index
#     unrated_items = user_ratings[user_ratings == 0].index
#     
#     recommendations = {}
#     for item in unrated_items:
#         score = 0
#         for liked_item in liked_items:
#             score += similarity_df.loc[item, liked_item]
#         recommendations[item] = score
#     
#     sorted_recommendations = sorted(recommendations.items(), 
#                                    key=lambda x: x[1], 
#                                    reverse=True)
#     return sorted_recommendations[:n_recommendations]
# ```
# </details>

# MARKDOWN ********************

# # 3) Calculate User-to-User Similarity #
# 
# Now let's try the **user-to-user** approach. Calculate similarity between users based on their rating patterns.
# 
# This time, we don't transpose the matrix because users are already rows.
# 
# Try using **Pearson correlation** instead of cosine similarity. Pearson correlation accounts for the fact that some users tend to rate everything high while others rate everything low.

# CELL ********************

# YOUR CODE HERE
# Calculate user-to-user similarity using Pearson correlation
# Note: We need to handle NaN values from zero ratings

def calculate_user_similarity(ratings_df):
    """
    Calculate user-to-user similarity using Pearson correlation.
    Handles zero ratings by treating them as missing values.
    """
    n_users = len(ratings_df)
    similarity_matrix = np.zeros((n_users, n_users))
    
    for i in range(n_users):
        for j in range(n_users):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Get ratings from both users
                user_i = ratings_df.iloc[i].values
                user_j = ratings_df.iloc[j].values
                
                # Find items both users have rated (non-zero)
                mask = (user_i > 0) & (user_j > 0)
                
                if mask.sum() > 1:  # Need at least 2 common ratings
                    # YOUR CODE HERE: Calculate Pearson correlation
                    corr, _ = ____
                    similarity_matrix[i, j] = corr if not np.isnan(corr) else 0
                else:
                    similarity_matrix[i, j] = 0
    
    return similarity_matrix

user_similarity = calculate_user_similarity(ratings_df)
user_similarity_df = pd.DataFrame(user_similarity, 
                                  index=users, 
                                  columns=users)

print("User-to-User Similarity Matrix:")
print(user_similarity_df.round(3))

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(user_similarity_df, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1)
plt.title('User Similarity Heatmap (Pearson Correlation)')
plt.tight_layout()
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# <details>
# <summary>Click here for Solution to Exercise 3</summary>
# 
# ```python
# corr, _ = pearsonr(user_i[mask], user_j[mask])
# ```
# </details>

# MARKDOWN ********************

# # 4) Make User-Based Recommendations #
# 
# Implement a function to recommend movies based on **user-to-user similarity**.
# 
# The algorithm should:
# 1. Find users most similar to the target user
# 2. Look at what those similar users rated highly
# 3. Recommend items the target user hasn't seen
# 4. Weight recommendations by user similarity

# CELL ********************

def recommend_user_based(user, ratings_df, user_similarity_df, n_recommendations=3, k_neighbors=3):
    """
    Recommend items based on user-to-user similarity.
    
    Parameters:
    - user: name of the user
    - ratings_df: user-item ratings matrix
    - user_similarity_df: user-user similarity matrix
    - n_recommendations: number of items to recommend
    - k_neighbors: number of similar users to consider
    
    Returns:
    - List of recommended items with predicted ratings
    """
    # YOUR CODE HERE
    # Step 1: Find k most similar users (excluding the user themselves)
    similar_users = ____
    
    # Step 2: Get target user's ratings
    user_ratings = ____
    
    # Step 3: Find unrated items
    unrated_items = ____
    
    # Step 4: Predict ratings for unrated items
    predictions = {}
    for item in unrated_items:
        # Calculate weighted average of similar users' ratings
        weighted_sum = 0
        similarity_sum = 0
        
        for similar_user in similar_users.index:
            rating = ratings_df.loc[similar_user, item]
            if rating > 0:  # Only consider if similar user rated this item
                similarity = similar_users[similar_user]
                # YOUR CODE HERE
                ____
        
        if similarity_sum > 0:
            predictions[item] = weighted_sum / similarity_sum
    
    # Step 5: Sort and return top N
    sorted_predictions = sorted(predictions.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
    return sorted_predictions[:n_recommendations]

# Test your function
user_to_recommend = 'Eve'
recommendations = recommend_user_based(user_to_recommend, ratings_df, user_similarity_df)

print(f"\nTop recommendations for {user_to_recommend} (user-based):")
for item, predicted_rating in recommendations:
    print(f"  {item}: predicted rating = {predicted_rating:.2f}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# <details>
# <summary>Click here for Solution to Exercise 4</summary>
# 
# ```python
# def recommend_user_based(user, ratings_df, user_similarity_df, n_recommendations=3, k_neighbors=3):
#     similar_users = user_similarity_df[user].sort_values(ascending=False)[1:k_neighbors+1]
#     user_ratings = ratings_df.loc[user]
#     unrated_items = user_ratings[user_ratings == 0].index
#     
#     predictions = {}
#     for item in unrated_items:
#         weighted_sum = 0
#         similarity_sum = 0
#         
#         for similar_user in similar_users.index:
#             rating = ratings_df.loc[similar_user, item]
#             if rating > 0:
#                 similarity = similar_users[similar_user]
#                 weighted_sum += similarity * rating
#                 similarity_sum += abs(similarity)
#         
#         if similarity_sum > 0:
#             predictions[item] = weighted_sum / similarity_sum
#     
#     sorted_predictions = sorted(predictions.items(), 
#                                key=lambda x: x[1], 
#                                reverse=True)
#     return sorted_predictions[:n_recommendations]
# ```
# </details>

# MARKDOWN ********************

# # 5) Compare Both Approaches #
# 
# Let's compare item-based and user-based recommendations for the same user:

# CELL ********************

# Compare both methods for Alice
test_user = 'Alice'

print(f"\n{'='*60}")
print(f"Recommendations for {test_user}")
print(f"{'='*60}")

print(f"\n{test_user}'s ratings:")
print(ratings_df.loc[test_user])

print(f"\n--- ITEM-BASED RECOMMENDATIONS ---")
item_recs = recommend_items(test_user, ratings_df, item_similarity_df, n_recommendations=3)
for item, score in item_recs:
    print(f"  {item}: score = {score:.3f}")

print(f"\n--- USER-BASED RECOMMENDATIONS ---")
user_recs = recommend_user_based(test_user, ratings_df, user_similarity_df, n_recommendations=3)
for item, rating in user_recs:
    print(f"  {item}: predicted rating = {rating:.2f}")

print(f"\n{'='*60}")
print("Analysis:")
print("- Item-based focuses on movies similar to what you liked")
print("- User-based recommends what similar users enjoyed")
print("- Different approaches can yield different recommendations!")
print(f"{'='*60}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Congratulations! #
# 
# You've successfully implemented both **Item-to-Item** and **User-to-User** collaborative filtering!
# 
# **Key Takeaways:**
# - Item-based CF is more stable and scalable for many users
# - User-based CF can capture more nuanced preferences
# - Both approaches suffer from cold start and sparsity problems
# - Real-world systems often combine multiple approaches
# 
# **Next Steps:**
# In the next lesson, you'll learn about **Collaborative Filtering with Matrix Factorization**, which can handle sparse data better and scale to millions of users and items!

