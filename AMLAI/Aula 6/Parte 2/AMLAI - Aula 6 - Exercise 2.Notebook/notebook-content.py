# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# # Natural Language Classification
# 
# You did such a great job for DeFalco's restaurant in the previous exercise that the chef has hired you for a new project.
# 
# The restaurant's menu includes an email address where visitors can give feedback about their food. 
# 
# The manager wants you to create a tool that automatically sends him all the negative reviews so he can fix them, while automatically sending all the positive reviews to the owner, so the manager can ask for a raise. 
# 
# You will first build a model to distinguish positive reviews from negative reviews using Yelp reviews because these reviews include a rating with each review. Your data consists of the text body of each review along with the star rating. Ratings with 1-2 stars count as "negative", and ratings with 4-5 stars are "positive". Ratings with 3 stars are "neutral" and have been dropped from the data.
# 
# Let's get started. First, run the next code cell.

# CELL ********************

pip install spacy 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.nlp.ex2 import *
print("\nSetup complete")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Step 1: Evaluate the Approach
# 
# Is there anything about this approach that concerns you? After you've thought about it, run the function below to see one point of view.

# CELL ********************

# Check your answer (Run this code cell to receive credit!)
step_1.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Step 2: Review Data and Create the model
# 
# Moving forward with your plan, you'll need to load the data. Here's some basic code to load data and split it into a training and validation set. Run this code.

# CELL ********************

def load_data(csv_file, split=0.9):
    data = pd.read_csv(csv_file)
    
    # Shuffle data
    train_data = data.sample(frac=1, random_state=7)
    
    texts = train_data.text.values
    labels = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)}
              for y in train_data.sentiment.values]
    split = int(len(train_data) * split)
    
    train_labels = [{"cats": labels} for labels in labels[:split]]
    val_labels = [{"cats": labels} for labels in labels[split:]]
    
    return texts[:split], train_labels, texts[split:], val_labels

train_texts, train_labels, val_texts, val_labels = load_data('../input/nlp-course/yelp_ratings.csv')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# You will use this training data to build a model. The code to build the model is the same as what you saw in the tutorial. So that is copied below for you.
# 
# First, run the cell below to look at a couple elements from your training data.

# CELL ********************

print('Texts from training data\n------')
print(train_texts[:2])
print('\nLabels from training data\n------')
print(train_labels[:2])


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# But because your data is different, there are **two lines in the modeling code cell that you'll need to change.** Can you figure out what they are? 
# 
# If you're not sure, take a second look at the data, and pay particular attention to the labels that should be fed to the text classifier.

# CELL ********************

import spacy

# Create an empty model
nlp = spacy.blank('en')

# Add the TextCategorizer to the empty model
textcat = nlp.add_pipe('textcat')

# Add labels to text classifier
textcat.add_label("ham")
textcat.add_label("spam")

# Check your answer
step_2.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
step_2.hint()
step_2.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Step 3: Train Function
# 
# Implement a function `train` that updates a model with training data. Most of this is general data munging, which we've filled in for you. 
# 
# Just add the one line of code necessary to update your model.

# CELL ********************

import random
from spacy.util import minibatch
from spacy.training.example import Example

def train(model, train_data, optimizer, batch_size=8):
    losses = {}
    random.seed(1)
    random.shuffle(train_data)
    
    # train_data is a list of tuples [(text0, label0), (text1, label1), ...]
    for batch in minibatch(train_data, size=batch_size):
        # Split batch into text and labels
        for text, labels in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, labels)
            # TODO: Update model with texts and labels
            ____
        
    return losses

# Check your answer
step_3.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
step_3.hint()
step_3.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Fix seed for reproducibility
spacy.util.fix_random_seed(1)
random.seed(1)

# This may take a while to run!
optimizer = nlp.begin_training()
train_data = list(zip(train_texts, train_labels))
losses = train(nlp, train_data, optimizer)
print(losses['textcat'])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# We can try this slightly trained model on some example text and look at the probabilities assigned to each label.

# CELL ********************

text = "This tea cup was full of holes. Do not recommend."
doc = nlp(text)
print(doc.cats)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# These probabilities look reasonable. Now you should turn them into an actual prediction.
# 
# # Step 4: Making Predictions
# 
# Implement a function `predict` that predicts the sentiment of text examples. 
# - First, tokenize the texts using `nlp.tokenizer()`. 
# - Then, pass those docs to the TextCategorizer which you can get from `nlp.get_pipe()`. 
# - Use the `textcat.predict()` method to get scores for each document, then choose the class with the highest score (probability) as the predicted class.

# CELL ********************

def predict(nlp, texts): 
    # Use the model's tokenizer to tokenize each input text
    docs = ____
    
    # Use textcat to get the scores for each doc
    ____
    
    # From the scores, find the class with the highest score/probability
    predicted_class = ____
    
    return predicted_class

# Check your answer
step_4.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
step_4.hint()
step_4.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

texts = val_texts[34:38]
predictions = predict(nlp, texts)

for p, t in zip(predictions, texts):
    print(f"{textcat.labels[p]}: {t} \n")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# It looks like your model is working well after going through the data just once. However you need to calculate some metric for the model's performance on the hold-out validation data.
# 
# # Step 5: Evaluate The Model
# 
# Implement a function that evaluates a `TextCategorizer` model. This function `evaluate` takes a model along with texts and labels. It returns the accuracy of the model, which is the number of correct predictions divided by all predictions.
# 
# First, use the `predict` method you wrote earlier to get the predicted class for each text in `texts`. Then, find where the predicted labels match the true "gold-standard" labels and calculate the accuracy.

# CELL ********************

def evaluate(model, texts, labels):
    """ Returns the accuracy of a TextCategorizer model. 
    
        Arguments
        ---------
        model: ScaPy model with a TextCategorizer
        texts: Text samples, from load_data function
        labels: True labels, from load_data function
    
    """
    # Get predictions from textcat model (using your predict method)
    predicted_class = ____
    
    # From labels, get the true class as a list of integers (POSITIVE -> 1, NEGATIVE -> 0)
    true_class = ____
    
    # A boolean or int array indicating correct predictions
    correct_predictions = ____
    
    # The accuracy, number of correct predictions divided by all predictions
    accuracy = ____
    
    return accuracy

# Check your answer
step_5.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Lines below will give you a hint or solution code
step_5.hint()
step_5.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

accuracy = evaluate(nlp, val_texts, val_labels)
print(f"Accuracy: {accuracy:.4f}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# With the functions implemented, you can train and evaluate in a loop.

# CELL ********************

# This may take a while to run!
n_iters = 5
for i in range(n_iters):
    losses = train(nlp, train_data, optimizer)
    accuracy = evaluate(nlp, val_texts, val_labels)
    print(f"Loss: {losses['textcat']:.3f} \t Accuracy: {accuracy:.3f}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Step 6: Keep Improving
# 
# You've built the necessary components to train a text classifier with spaCy. What could you do further to optimize the model?
# 
# Run the next line to check your answer.

# CELL ********************

# Check your answer (Run this code cell to receive credit!)
step_6.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Keep Going
# 
# The next step is a big one. See how you can **[represent tokens as vectors that describe their meaning]**, and plug those into your machine learning models.
