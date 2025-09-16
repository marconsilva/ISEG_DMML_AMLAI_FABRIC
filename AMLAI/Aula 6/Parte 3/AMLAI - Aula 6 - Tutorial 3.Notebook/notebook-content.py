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
# META     }
# META   }
# META }

# MARKDOWN ********************

# # Word Embeddings
# 
# You know at this point that machine learning on text requires that you first represent the text numerically. So far, you've done this with bag of words representations. But you can usually do better with word embeddings.
# 
# **Word embeddings** (also called word vectors) represent each word numerically in such a way that the vector corresponds to how that word is used or what it means. Vector encodings are learned by considering the context in which the words appear. Words that appear in similar contexts will have similar vectors. For example, vectors for "leopard", "lion", and "tiger" will be close together, while they'll be far away from "planet" and "castle".
# 
# Even cooler, relations between words can be examined with mathematical operations. Subtracting the vectors for "man" and "woman" will return another vector. If you add that to the vector for "king" the result is close to the vector for "queen."
# 
# ![Word vector examples](https://www.tensorflow.org/images/linear-relationships.png)
# 
# These vectors can be used as features for machine learning models. Word vectors will typically improve the performance of your models above bag of words encoding. spaCy provides embeddings learned from a model called Word2Vec. You can access them by loading a large language model like `en_core_web_lg`. Then they will be available on tokens from the `.vector` attribute.

# CELL ********************

pip install spacy

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

!python -m spacy download en_core_web_lg 
!python -m spacy download en_core_web_sm


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import numpy as np
import spacy
import mlflow
mlflow.autolog(disable=True)

# Need to load the large model to get the vectors
nlp = spacy.load('en_core_web_lg')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Disabling other pipes because we don't need them and it'll speed up this part a bit
text = "These vectors can be used as features for machine learning models."
with nlp.disable_pipes():
    vectors = np.array([token.vector for token in  nlp(text)])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

vectors.shape

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# These are 300-dimensional vectors, with one vector for each word. However, we only have document-level labels and our models won't be able to use the word-level embeddings. So, you need a vector representation for the entire document. 
# 
# There are many ways to combine all the word vectors into a single document vector we can use for model training. A simple and surprisingly effective approach is simply averaging the vectors for each word in the document. Then, you can use these document vectors for modeling.
# 
# spaCy calculates the average document vector which you can get with `doc.vector`. Here is an example loading the spam data and converting it to document vectors.

# CELL ********************

import pandas as pd

# Loading the spam data
# ham is the label for non-spam messages
spam = pd.read_csv('/lakehouse/default/Files/AMLAI_Aula6/spam.csv')

with nlp.disable_pipes():
    doc_vectors = np.array([nlp(text).vector for text in spam.text])
    
doc_vectors.shape

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Classification Models
# 
# With the document vectors, you can train scikit-learn models, xgboost models, or any other standard approach to modeling.

# CELL ********************

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(doc_vectors, spam.label,
                                                    test_size=0.1, random_state=1)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Here is an example using [support vector machines (SVMs)](https://scikit-learn.org/stable/modules/svm.html#svm). Scikit-learn provides an SVM classifier `LinearSVC`. This works similar to other scikit-learn models.

# CELL ********************

from sklearn.svm import LinearSVC

# Set dual=False to speed up training, and it's not needed
svc = LinearSVC(random_state=1, dual=False, max_iter=10000)
svc.fit(X_train, y_train)
print(f"Accuracy: {svc.score(X_test, y_test) * 100:.3f}%", )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Document Similarity
# 
# Documents with similar content generally have similar vectors. So you can find similar documents by measuring the similarity between the vectors. A common metric for this is the **cosine similarity** which measures the angle between two vectors, $\mathbf{a}$ and $\mathbf{b}$.
# 
# $$
# \cos \theta = \frac{\mathbf{a}\cdot\mathbf{b}}{\| \mathbf{a} \| \, \| \mathbf{b} \|}
# $$
# 
# This is the dot product of $\mathbf{a}$ and $\mathbf{b}$, divided by the magnitudes of each vector. The cosine similarity can vary between -1 and 1, corresponding complete opposite to perfect similarity, respectively. To calculate it, you can use [the metric from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) or write your own function.

# CELL ********************

def cosine_similarity(a, b):
    return a.dot(b)/np.sqrt(a.dot(a) * b.dot(b))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

a = nlp("REPLY NOW FOR FREE TEA").vector
b = nlp("According to legend, Emperor Shen Nung discovered tea when leaves from a wild tree blew into his pot of boiling water.").vector
cosine_similarity(a, b)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Your Turn
# Word embeddings are incredibly powerful. You know know enough to apply embeddings to **[improve your models and find similar documents]**.

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
