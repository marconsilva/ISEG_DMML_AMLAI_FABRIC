# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# # Vectorizing Language
# 
# Embeddings are both conceptually clever and practically effective. 
# 
# So let's try them for the sentiment analysis model you built for the restaurant. Then you can find the most similar review in the data set given some example text. It's a task where you can easily judge for yourself how well the embeddings work.

# CELL ********************

!python -m spacy download en_core_web_lg 
!python -m spacy download en_core_web_sm


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

pip install spacy

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.nlp.ex3 import *
print("\nSetup complete")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Load the large model to get the vectors
nlp = spacy.load('en_core_web_lg')

review_data = pd.read_csv('../input/nlp-course/yelp_ratings.csv')
review_data.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Here's an example of loading some document vectors. 
# 
# Calculating 44,500 document vectors takes about 20 minutes, so we'll get only the first 100. To save time, we'll load pre-saved document vectors for the hands-on coding exercises.

# CELL ********************

reviews = review_data[:100]
# We just want the vectors so we can turn off other models in the pipeline
with nlp.disable_pipes():
    vectors = np.array([nlp(review.text).vector for idx, review in reviews.iterrows()])
    
vectors.shape

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# The result is a matrix of 100 rows and 300 columns. 
# 
# Why 100 rows?
# Because we have 1 row for each column.
# 
# Why 300 columns?
# This is the same length as word vectors. See if you can figure out why document vectors have the same length as word vectors (some knowledge of linear algebra or vector math would be needed to figure this out).

# MARKDOWN ********************

# Go ahead and run the following cell to load in the rest of the document vectors.

# CELL ********************

# Loading all document vectors from file
vectors = np.load('../input/nlp-course/review_vectors.npy')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # 1) Training a Model on Document Vectors
# 
# Next you'll train a `LinearSVC` model using the document vectors. It runs pretty quick and works well in high dimensional settings like you have here.
# 
# After running the LinearSVC model, you might try experimenting with other types of models to see whether it improves your results.

# CELL ********************

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(vectors, review_data.sentiment, 
                                                    test_size=0.1, random_state=1)

# Create the LinearSVC model
model = LinearSVC(random_state=1, dual=False)
# Fit the model
____

# Uncomment and run to see model accuracy
# print(f'Model test accuracy: {model.score(X_test, y_test)*100:.3f}%')

# Uncomment to check your work
#q_1.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
#_COMMENT_IF(PROD)_
q_1.hint()
#_COMMENT_IF(PROD)_
q_1.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#%%RM_IF(PROD)%%
model = LinearSVC(random_state=1, dual=False)
model.fit(X_train, y_train)
q_1.assert_check_passed()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Scratch space in case you want to experiment with other models

#second_model = ____
#second_model.fit(X_train, y_train)
#print(f'Model test accuracy: {second_model.score(X_test, y_test)*100:.3f}%')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Document Similarity
# 
# For the same tea house review, find the most similar review in the dataset using cosine similarity.
# 
# # 2) Centering the Vectors
# 
# Sometimes people center document vectors when calculating similarities. That is, they calculate the mean vector from all documents, and they subtract this from each individual document's vector. Why do you think this could help with similarity metrics?
# 
# Run the following line after you've decided your answer.

# CELL ********************

# Check your answer (Run this code cell to receive credit!)
q_2.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # 3) Find the most similar review
# 
# Given an example review below, find the most similar document within the Yelp dataset using the cosine similarity.

# CELL ********************

review = """I absolutely love this place. The 360 degree glass windows with the 
Yerba buena garden view, tea pots all around and the smell of fresh tea everywhere 
transports you to what feels like a different zen zone within the city. I know 
the price is slightly more compared to the normal American size, however the food 
is very wholesome, the tea selection is incredible and I know service can be hit 
or miss often but it was on point during our most recent visit. Definitely recommend!

I would especially recommend the butternut squash gyoza."""

def cosine_similarity(a, b):
    return np.dot(a, b)/np.sqrt(a.dot(a)*b.dot(b))

review_vec = nlp(review).vector

## Center the document vectors
# Calculate the mean for the document vectors, should have shape (300,)
vec_mean = vectors.mean(axis=0)
# Subtract the mean from the vectors
centered = ____

# Calculate similarities for each document in the dataset
# Make sure to subtract the mean from the review vector
sims = ____

# Get the index for the most similar document
most_similar = ____

# Uncomment to check your work
#q_3.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
#_COMMENT_IF(PROD)_
q_3.hint()
#_COMMENT_IF(PROD)_
q_3.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#%%RM_IF(PROD)%%
review_vec = nlp(review).vector

## Center the document vectors
# Calculate the mean for the document vectors
vec_mean = vectors.mean(axis=0)
# Subtract the mean from the vectors
centered = vectors - vec_mean

# Calculate similarities for each document in the dataset
# Make sure to subtract the mean from the review vector
sims = np.array([cosine_similarity(review_vec - vec_mean, vec) for vec in centered])

# Get the index for the most similar document
most_similar = sims.argmax()
q_3.assert_check_passed()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print(review_data.iloc[most_similar].text)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Even though there are many different sorts of businesses in our Yelp dataset, you should have found another tea shop. 
# 
# # 4) Looking at similar reviews
# 
# If you look at other similar reviews, you'll see many coffee shops. Why do you think reviews for coffee are similar to the example review which mentions only tea?

# CELL ********************

# Check your answer (Run this code cell to receive credit!)
q_4.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Congratulations!
# 
# You've finished the first part of NLP tutorial. It's an exciting field that will help you make use of vast amounts of data you didn't know how to work with before.
# 
# Keep going
