# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# # Intro
# 
# Data comes in many different forms: time stamps, sensor readings, images, categorical labels, and so much more. But text is still some of the most valuable data out there for those who know how to use it.  
# 
# In this tutorial about **Natural Language Processing (NLP)**, you will use the leading NLP library (spaCy) to take on some of the most important tasks in working with text. 
# 
# By the end, you will be able to use spaCy for:
# 
# * Basic text processing and pattern matching
# * Building machine learning models with text
# * Representing text with word embeddings that numerically capture the meaning of words and documents

# MARKDOWN ********************

# ## NLP with spaCy
# 
# spaCy is the leading library for NLP, and it has quickly become one of the most popular Python frameworks. Most people find it intuitive, and it has excellent [documentation](https://spacy.io/usage).
# 
# spaCy relies on **models** that are language-specific and come in different sizes. You can load a spaCy model with `spacy.load`. 
# 
# For example, here's how you would load the English language model.

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

import spacy
from spacy.lang.en.examples import sentences 
nlp = spacy.load('en_core_web_sm')


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# With the model loaded, you can process text like this:

# CELL ********************

doc = nlp("Tea is healthy and calming, don't you think?")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# There's a lot you can do with the `doc` object you just created.
# 
# # Tokenizing
# 
# This returns a document object that contains **tokens**. A token is a unit of text in the document, such as individual words and punctuation. SpaCy splits contractions like "don't" into two tokens, "do" and "n't". You can see the tokens by iterating through the document.

# CELL ********************

for token in doc:
    print(token)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Iterating through a document gives you token objects. Each of these tokens comes with additional information. In most cases, the important ones are `token.lemma_` and `token.is_stop`.
# 
# # Text preprocessing
# 
# There are a few types of preprocessing to improve how we model with words. The first is "lemmatizing."
# The "lemma" of a word is its base form.  For example, "walk" is the lemma of the word "walking". So, when you lemmatize the word walking, you would convert it to walk.
# 
# It's also common to remove stopwords. Stopwords are words that occur frequently in the language and don't contain much information. English  stopwords include "the", "is", "and", "but", "not". 
# 
# With a spaCy token, `token.lemma_` returns the lemma, while `token.is_stop` returns a boolean `True` if the token is a stopword (and `False` otherwise).

# CELL ********************

print(f"Token \t\tLemma \t\tStopword".format('Token', 'Lemma', 'Stopword'))
print("-"*40)
for token in doc:
    print(f"{str(token)}\t\t{token.lemma_}\t\t{token.is_stop}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Why are lemmas and identifying stopwords important? Language data has a lot of noise mixed in with informative content. In the sentence above, the important words are tea, healthy and calming. Removing stop words might help the predictive model focus on relevant words. Lemmatizing similarly helps by combining multiple forms of the same word into one base form ("calming", "calms", "calmed" would all change to "calm").
# 
# However, lemmatizing and dropping stopwords might result in your models performing worse. So you should treat this preprocessing as part of your hyperparameter optimization process.

# MARKDOWN ********************

# # Pattern Matching
# 
# Another common NLP task is matching tokens or phrases within chunks of text or whole documents. You can do pattern matching with regular expressions, but spaCy's matching capabilities tend to be easier to use.
# 
# To match individual tokens, you create a `Matcher`. When you want to match a list of terms, it's easier and more efficient to use `PhraseMatcher`. For example, if you want to find where different smartphone models show up in some text, you can create patterns for the model names of interest. First you create the `PhraseMatcher` itself.

# CELL ********************

from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# The matcher is created using the vocabulary of your model. Here we're using the small English model you loaded earlier. Setting `attr='LOWER'` will match the phrases on lowercased text. This provides case insensitive matching.
# 
# Next you create a list of terms to match in the text. The phrase matcher needs the patterns as document objects. The easiest way to get these is with a list comprehension using the `nlp` model.

# CELL ********************

terms = ['Galaxy Note', 'iPhone 11', 'iPhone XS', 'Google Pixel']
patterns = [nlp(text) for text in terms]
matcher.add("TerminologyList", patterns)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Then you create a document from the text to search and use the phrase matcher to find where the terms occur in the text.

# CELL ********************

text_doc = nlp("Glowing review overall, and some really interesting side-by-side "
               "photography tests pitting the iPhone 11 Pro against the "
               "Galaxy Note 10 Plus and last year’s iPhone XS and Google Pixel 3.") 
matches = matcher(text_doc)
print(matches)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# The matches here are a tuple of the match id and the positions of the start and end of the phrase.

# CELL ********************

match_id, start, end = matches[0]
print(nlp.vocab.strings[match_id], text_doc[start:end])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Your Turn
# Now that you've seen a few uses of SpaCy for NLP, it's your turn to try it to **[analyze Yelp reviews]**.
