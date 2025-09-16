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

# Convolutional Neural Networks
# ======
# 
# Convolutional neural networks (CNNs) are a class of deep neural networks, most commonly used in computer vision applications.
# 
# Convolutional refers the network pre-processing data for you - traditionally this pre-processing was performed by data scientists. The neural network can learn how to do pre-processing *itself* by applying filters for things such as edge detection.

# MARKDOWN ********************

# Step 1
# -----
# 
# In this exercise we will train a CNN to recognise handwritten digits, using the MNIST digit dataset.
# 
# This is a very common exercise and data set to learn from.
# 
# Let's start by loading our dataset and setting up our train, validation, and test sets.
# 
# #### Run the code below to import our required libraries and set up the graphing features.

# CELL ********************

# Run this!
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
import mlflow
mlflow.autolog(disable=True)
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
print('keras using %s backend'%keras.backend.backend())
import matplotlib.pyplot as graph
%matplotlib inline
graph.rcParams['figure.figsize'] = (15,5)
graph.rcParams["font.family"] = 'DejaVu Sans'
graph.rcParams["font.size"] = '12'
graph.rcParams['image.cmap'] = 'rainbow'

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Here we will add the code for split into train , valid and test

# CELL ********************

# Here we import the dataset, and split it into the training, validation, and test sets.
from tensorflow.keras.datasets import mnist

# This is our training data, with 6400 samples.
###
###
train_X = mnist.load_data()[0][0][:6400].astype('float32')
train_Y = mnist.load_data()[0][1][:6400]
###

# This is our validation data, with 1600 samples.
###
###
valid_X = mnist.load_data()[1][0][:1600].astype('float32')
valid_Y = mnist.load_data()[1][1][:1600]
###

# This is our test data, with 2000 samples.
###
###
test_X = mnist.load_data()[1][0][-2000:].astype('float32')
test_Y = mnist.load_data()[1][1][-2000:]
###

print('train_X:', train_X.shape, end = '')
print(', train_Y:', train_Y.shape)
print('valid_X:', valid_X.shape, end = '')
print(', valid_Y:', valid_Y.shape)
print('test_X:', test_X.shape, end = '')
print(', test_Y:', test_Y.shape)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# So we have 6400 training samples, 1600 validation samples, and 2000 test samples.
# 
# Each sample is an greyscale image - 28 pixels wide and 28 pixels high. Each pixel is really a number from 0 to 255 - 0 being fully black, 255 being fully white. When we graph the 28x28 numbers, we can see the image.
# 
# Let's have a look at one of our samples.


# CELL ********************

###
graph.imshow(train_X[0], cmap = 'gray', interpolation = 'nearest')
###

graph.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Step 2
# ---
# 
# The neural network will use the 28x28 values of each image to predict what each image represents.
# 
# As each value is between 0 and 255, we'll scale the values down by dividing by 255 (this makes it faster for the Neural Network to train).
# 
# We need to reshape our data to get it working well with our neural network. 


# CELL ********************

# First off, let's reshape our X sets so that they fit the convolutional layers.

# This gets the image dimensions - 28
dim = train_X[0].shape[0]

###
###
train_X = train_X.reshape(train_X.shape[0], dim, dim, 1)
valid_X = valid_X.reshape(valid_X.shape[0], dim, dim, 1)
test_X = test_X.reshape(test_X.shape[0], dim, dim, 1)
###

# Next up - feature scaling.
# We scale the values so they are between 0 and 1, instead of 0 and 255.

###
###
train_X = train_X/255
valid_X = valid_X/255
test_X = test_X/255
###


# Now we print the label for the first example
print(train_Y[0])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Expected output:  
# `5`
# 
# The label is a number - the number we see when we view the image.
# 
# We need represent this number as a one-hot vector, so the neural network knows it is a category.
# 
# Keras can convert these labels into one-hot vectors easily with the function - `to_categorical`
# 


# CELL ********************

from tensorflow.keras.utils import to_categorical

###
###
train_Y = keras.utils.to_categorical(train_Y, 10)
valid_Y = keras.utils.to_categorical(valid_Y, 10)
test_Y = keras.utils.to_categorical(test_Y, 10)
###

# 10 being the number of categories (numbers 0 to 9)

print(train_Y[0])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Expected output:  
# `[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]`
# 
# Step 3
# -----
# 
# All ready! Time to build another neural network.
# Starting with Sequential()


# CELL ********************

# Sets a randomisation seed for replicatability.
np.random.seed(6)

###
###
model = Sequential()
###

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# The __Convolutional__ in Convolutional Neural Networks refers the pre-processing the network can do itself.
# 


# CELL ********************

###
###
model.add(Conv2D(28, kernel_size = (3, 3), activation = 'relu', input_shape = (dim, dim, 1)))
model.add(Conv2D(56, (3, 3), activation = 'relu'))
###

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Next up we'll:
# * Add pooling layers.
# * Apply dropout.
# * Flatten the data to a vector (the output of step 2 is a vector).
# 


# CELL ********************

# Pooling layers help speed up training time and make features it detects more robust.
# They act by downsampling the data - reducing the data size and complexity.

###
###
model.add(MaxPooling2D(pool_size = (2, 2)))
###

# Dropout is a technique to help prevent overfitting
# It makes nodes 'dropout' - turning them off randomly.

###
###
model.add(Dropout(0.125))
###


###
###
model.add(Flatten())
###

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# In the end we add a Dense layer with softmax activation with the value 10 and run the code.

# CELL ********************

# Dense layers perform classification - we have extracted the features with the convolutional pre-processing
model.add(Dense(128, activation='relu'))

# More dropout!
model.add(Dropout(0.25))

# Next is our output layer
# Softmax outputs the probability for each category
###
###
model.add(Dense(10, activation=tf.nn.softmax))
###

# And finally, we compile.
model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Step 4

# CELL ********************

###
###
training_stats = model.fit(train_X, train_Y, batch_size = 128, epochs = 12, verbose = 1, validation_data = (valid_X, valid_Y))
###

###
###
evaluation = model.evaluate(test_X, test_Y, verbose=0)
###

print('Test Set Evaluation: loss = %0.6f, accuracy = %0.2f' %(evaluation[0], 100 * evaluation[1]))

# We can plot our training statistics to see how it developed over time
accuracy, = graph.plot(training_stats.history['accuracy'], label = 'Accuracy')
training_loss, = graph.plot(training_stats.history['loss'], label = 'Training Loss')
graph.legend(handles = [accuracy, training_loss])
loss = np.array(training_stats.history['loss'])
xp = np.linspace(0,loss.shape[0],10 * loss.shape[0])
graph.plot(xp, np.full(xp.shape, 1), c = 'k', linestyle = ':', alpha = 0.5)
graph.plot(xp, np.full(xp.shape, 0), c = 'k', linestyle = ':', alpha = 0.5)
graph.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Step 5
# 
# Let's test it on a new sample that it hasn't seen, and see how it classifies it!


# CELL ********************

###
# REPLACE THE <addNumber> WITH ANY NUMBER BETWEEN 0 AND 1999
###
test_sample_random = 50

sample = test_X[test_sample_random].reshape(dim, dim)
###

graph.imshow(sample, cmap = 'gray', interpolation = 'nearest')
graph.show()

prediction = model.predict(sample.reshape(1, dim, dim, 1))
print('prediction: %i' %(np.argmax(prediction)))
print('real value', np.argmax(test_Y[test_sample_random]))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# How is the prediction? Does it look right?

# MARKDOWN ********************

# Conclusion
# ------
# 
# Congratulations! We've built a convolutional neural network that is able to recognise handwritten digits with very high accuracy.
# 
# CNN's are very complex - you're not expected to understand everything (or most things) we covered here. They take a lot of time and practise to properly understand each aspect of them.
# 
# Here we used:  
# * __Feature scaling__ - reducing the range of the values. This helps improve training time.
# * __Convolutional layers__ - network layers that pre-process the data for us. These apply filters to extract features for the neural network to analyze.
# * __Pooling layers__ - part of the Convolutional layers. They apply filters downsample the data - extracting features.
# * __Dropout__ - a regularization technique to help prevent overfitting.
# * __Dense layers__ - neural network layers which perform classification on the features extracted by the convolutional layers and downsampled by the pooling layers.
# * __Softmax__ - an activation function which outputs the probability for each category.
