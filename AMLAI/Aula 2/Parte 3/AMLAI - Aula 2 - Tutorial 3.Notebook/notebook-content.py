# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# # AMLAI - Aula 2 - Tutorial 3: Introduction to Neural Networks #
#
# Neural networks combine layers of simple transformations to learn complex relationships from data. In this introduction, you will see how dense layers are stacked into a regression model and how activation functions add nonlinearity.
#
# ## Stacking Dense Layers ##
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
# Continue with the neural-network lessons in **AMLAI - Aula 3**, where the tutorials and exercises build on this introduction.
