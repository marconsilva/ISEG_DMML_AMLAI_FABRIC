# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "jupyter",
# META     "jupyter_kernel_name": "python3.11"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "81cbac54-cfa3-495b-ad48-b44a92bb72fb",
# META       "default_lakehouse_name": "DataScienceLearnLakehouse",
# META       "default_lakehouse_workspace_id": "a677a3bf-5fb2-455e-abaa-9e850bde3e1a",
# META       "known_lakehouses": [
# META         {
# META           "id": "81cbac54-cfa3-495b-ad48-b44a92bb72fb"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# In this tutorial, you will learn how to use **cross-validation** for better measures of model performance.
# 
# # Introduction
# 
# Machine learning is an iterative process. 
# 
# You will face choices about what predictive variables to use, what types of models to use, what arguments to supply to those models, etc. So far, you have made these choices in a data-driven way by measuring model quality with a validation (or holdout) set.  
# 
# But there are some drawbacks to this approach.  To see this, imagine you have a dataset with 5000 rows.  You will typically keep about 20% of the data as a validation dataset, or 1000 rows.  But this leaves some random chance in determining model scores.  That is, a model might do well on one set of 1000 rows, even if it would be inaccurate on a different 1000 rows.  
# 
# At an extreme, you could imagine having only 1 row of data in the validation set. If you compare alternative models, which one makes the best predictions on a single data point will be mostly a matter of luck!
# 
# In general, the larger the validation set, the less randomness (aka "noise") there is in our measure of model quality, and the more reliable it will be.  Unfortunately, we can only get a large validation set by removing rows from our training data, and smaller training datasets mean worse models!


# MARKDOWN ********************

# # What is cross-validation?
# 
# In **cross-validation**, we run our modeling process on different subsets of the data to get multiple measures of model quality. 
# 
# For example, we could begin by dividing the data into 5 pieces, each 20% of the full dataset.  In this case, we say that we have broken the data into 5 "**folds**".  
# 
# ![tut5_crossval](https://storage.googleapis.com/kaggle-media/learn/images/9k60cVA.png)
# 
# Then, we run one experiment for each fold:
# - In **Experiment 1**, we use the first fold as a validation (or holdout) set and everything else as training data. This gives us a measure of model quality based on a 20% holdout set.  
# - In **Experiment 2**, we hold out data from the second fold (and use everything except the second fold for training the model). The holdout set is then used to get a second estimate of model quality.
# - We repeat this process, using every fold once as the holdout set.  Putting this together, 100% of the data is used as holdout at some point, and we end up with a measure of model quality that is based on all of the rows in the dataset (even if we don't use all rows simultaneously).
# 
# # When should you use cross-validation?
# 
# Cross-validation gives a more accurate measure of model quality, which is especially important if you are making a lot of modeling decisions.  However, it can take longer to run, because it estimates multiple models (one for each fold).  
# 
# So, given these tradeoffs, when should you use each approach?
# - _For small datasets_, where extra computational burden isn't a big deal, you should run cross-validation.
# - _For larger datasets_, a single validation set is sufficient.  Your code will run faster, and you may have enough data that there's little need to re-use some of it for holdout.
# 
# There's no simple threshold for what constitutes a large vs. small dataset.  But if your model takes a couple minutes or less to run, it's probably worth switching to cross-validation.  
# 
# Alternatively, you can run cross-validation and see if the scores for each experiment seem close.  If each experiment yields the same results, a single validation set is probably sufficient.


# MARKDOWN ********************

# # Example
# 
# We'll work with the same data as in the previous tutorial.  We load the input data in `X` and the output data in `y`.

# CELL ********************

import pandas as pd

# Read the data
data = pd.read_csv('/lakehouse/default/Files/DMML_Aula6/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Then, we define the steps that first uses an imputer to fill in missing values and a we use Ridge, a linear model with L2 regularization to make predictions.  
# 


# MARKDOWN ********************

# We obtain the cross-validation scores with the [`cross_val_score()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) function from scikit-learn.  We set the number of folds with the `cv` parameter.

# CELL ********************

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# Assuming X and y are already defined

# Step 1: Impute missing values
imputer = SimpleImputer()
X_imputed = imputer.fit_transform(X)

# Step 2: Define the model
model = Ridge(random_state=42)


# Step 3: Evaluate the model using cross-validation
# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(model, X_imputed, y, cv=5, scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# The `scoring` parameter chooses a measure of model quality to report: in this case, we chose negative mean absolute error (MAE).  The docs for scikit-learn show a [list of options](http://scikit-learn.org/stable/modules/model_evaluation.html).  
# 
# It is a little surprising that we specify *negative* MAE. Scikit-learn has a convention where all metrics are defined so a high number is better.  Using negatives here allows them to be consistent with that convention, though negative MAE is almost unheard of elsewhere. 
# 
# We typically want a single measure of model quality to compare alternative models.  So we take the average across experiments.

# CELL ********************

print("Average MAE score (across experiments):")
print(scores.mean())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Conclusion
# 
# Using cross-validation yields a much better measure of model quality, with the added benefit of cleaning up our code: note that we no longer need to keep track of separate training and validation sets.  So, especially for small datasets, it's a good improvement!
# 
# # Your Turn
# 
# Put your new skills to work in the **[next exercise]** notebook!
