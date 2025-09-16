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

# # Introduction
# 
# In the RNN lesson you have worked out with tesla stock, now we will be working with google stock price

# MARKDOWN ********************

# We will be doing the following 
# 
# ## Implementing Recurrent Neural Network with Keras
# * Loading and Preprocessing Data
# * Create RNN Model
# * Predictions and Visualising RNN Model

# MARKDOWN ********************

# <a id="31"></a>
# ### Loading and Preprocessing Data

# CELL ********************

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Run the cell bellow for finalizing set up

# CELL ********************

# Set up code checking
import os
from learntools.core import binder
binder.bind(globals())
from learntools.RNN_LSTM.ex4 import *
print("Setup Complete")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Importing the training set
dataset_train = pd.read_csv('/lakehouse/default/Files/AMLAI_Aula6/Google_Stock_Price_Train.csv')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

dataset_train.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

train = dataset_train.loc[:, ["Open"]].values
train

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
train_scaled = scaler.fit_transform(train)
train_scaled


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

plt.plot(train_scaled)
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Step 1 : Create data structure
# 
# Create a data structure with 50 timesteps and 1 output, somehow you should include a for loop up to 1258. X_train and y_train should be arrays.
# The picture bellow may help you
# 


# CELL ********************

# Creating a data structure with 50 timesteps and 1 output
X_train = []
y_train = []

timesteps = 50
for i in range(timesteps, 1258):
    X_train.append(train_scaled[i-timesteps:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


step_1.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

timesteps = ______
X_train = ________
y_train = ________
step_1.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ![50 timesteps and 1 output.png](attachment:7a3f50ef-79dd-4e71-be96-d22a7468afeb.png)

# ATTACHMENTS ********************

# ATTA {
# ATTA   "7a3f50ef-79dd-4e71-be96-d22a7468afeb.png": {
# ATTA     "image/png": "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"
# ATTA   }
# ATTA }

# CELL ********************

step_1.hint()
step_1.solution()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

y_train

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Step 2: Create RNN Model
# 
# Create an RNN called regressor with the following points in order
# - Sequential
# - SimpleRNN With units = 50,activation='tanh', return_sequences = True
# - Dropout 0.2
# - SimpleRNN With units = 50,activation='tanh', return_sequences = True
# - Dropout 0.2
# - SimpleRNN With units = 50,activation='tanh', return_sequences = True
# - Dropout 0.2
# - SimpleRNN With units = 50
# - Dense with units = 1
# 
# Finally compile with 
# optimizer = 'adam' and loss = 'mean_squared_error'
# 


# CELL ********************

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

step_2.check()




# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout

regressor = _________

step_2.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

step_2.hint()
step_2.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#We fit here the model with the training data
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# <a id="33"></a>
# ### Predictions and Visualising RNN Model

# CELL ********************

# Getting the real stock price of 2017
dataset_test = pd.read_csv('/lakehouse/default/Files/AMLAI_Aula6/Google_Stock_Price_Test.csv')
dataset_test.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

real_stock_price = dataset_test.loc[:, ["Open"]].values
real_stock_price

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - timesteps:].values.reshape(-1,1)
inputs = scaler.transform(inputs)  # min max scaler
inputs

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

X_test = []
for i in range(timesteps, 70):
    X_test.append(inputs[i-timesteps:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
# epoch = 250 daha güzel sonuç veriyor.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # Keep Going
# 
# You have completed the RNN topic, lets continue learning 
