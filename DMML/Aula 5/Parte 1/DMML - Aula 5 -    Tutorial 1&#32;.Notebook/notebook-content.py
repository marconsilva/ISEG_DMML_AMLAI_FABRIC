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

# # Introduction
# 
# 
# Linear Regression is one of the common and popular algorithm in Machine Learning. In fact, typically this would be the first algorithm that you will encounter while learning Machine Learning.
# Linear regression as the name suggests is a model that assumes a linear relationship between independent variable (x) and the dependent or target variable (y). 
# 
# 


# MARKDOWN ********************

# # Linear Regression Model
# 
# Linear regression is a fundamental model in machine learning used for predicting a continuous output variable based on input features. The model function for linear regression is represented as:
# 
# $$f_{w,b}(x) = wx + b$$
# 
# In this equation, $f_{w,b}(x)$ represents the predicted output, $w$ is the weight parameter, $b$ is the bias parameter, and $x$ is the input feature.
# 
# ## Model Training
# 
# To train a linear regression model, we aim to find the best values for the parameters $(w, b)$ that best fit our dataset.
# 
# ### Forward Pass
# 
# The forward pass is a step where we compute the linear regression output for the input data $X$ using the current weights and biases. It's essentially applying our model to the input data.
# 
# ### Cost Function
# 
# The cost function is used to measure how well our model is performing. It quantifies the difference between the predicted values and the actual values in our dataset. The cost function is defined as:
# 
# $$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m}(f_{w,b}(x^{(i)}) - y^{(i)})^2$$
# 
# Here, $J(w, b)$ is the cost, $m$ is the number of training examples, $x^{(i)}$ is the input data for the $i$-th example, $y^{(i)}$ is the actual output for the $i$-th example, and $w$ and $b$ are the weight and bias parameters, respectively.
# 
# ### Backward Pass (Gradient Computation)
# 
# The backward pass computes the gradients of the cost function with respect to the weights and biases. These gradients are crucial for updating the model parameters during training. The gradient formulas are as follows:
# 
# $$
# \frac{\partial J(w,b)}{\partial b} = \frac{1}{m} \sum_{i=0}^{m-1} (f_{w,b}(X^{(i)}) - y^{(i)})
# $$
# 
# $$
# \frac{\partial J(w,b)}{\partial w} = \frac{1}{m} \sum_{i=0}^{m-1} (f_{w,b}(X^{(i)}) - y^{(i)})X^{(i)}
# $$
# 
# ## Training Process
# 
# The training process involves iteratively updating the weights and biases to minimize the cost function. This is typically done through an optimization algorithm like gradient descent. The update equations for parameters are:
# 
# $$w \leftarrow w - \alpha \frac{\partial J}{\partial w}$$
# 
# $$b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$
# 
# Here, $\alpha$ represents the learning rate, which controls the step size during parameter updates.
# 
# By iteratively performing the forward pass, computing the cost, performing the backward pass, and updating the parameters, the model learns to make better predictions and fit the data.
# 
# As you can see in the picture bellow, we will apply the gradient descent and this will optimize and train our model:


# MARKDOWN ********************

# ![grad.png](attachment:720bfb0c-c541-4921-bfe3-f25ae4711b4a.png)

# ATTACHMENTS ********************

# ATTA {
# ATTA   "720bfb0c-c541-4921-bfe3-f25ae4711b4a.png": {
# ATTA     "image/png": "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"
# ATTA   }
# ATTA }

# MARKDOWN ********************

# ## Pratical Example 
# 
# Let's now apply this in practice. 


# MARKDOWN ********************

# # 💾 Data
# 
# We are going to use the `USA_Housing` dataset. Since house price is a continuous variable, this is a regression problem. The data contains the following columns:
# 
# * '`Avg. Area Income`': Avg. The income of residents of the city house is located in.
# * '`Avg. Area House Age`': Avg Age of Houses in the same city
# * '`Avg. Area Number of Rooms`': Avg Number of Rooms for Houses in the same city
# * '`Avg. Area Number of Bedrooms`': Avg Number of Bedrooms for Houses in the same city
#  * '`Area Population`': The population of city house is located in
# * '`Price`': Price that the house sold at
# * '`Address`': Address for the house


# MARKDOWN ********************

# # 📤 Import Libraries

# CELL ********************

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
mlflow.autolog(disable=True)

%matplotlib inline

# sns.set_style("whitegrid")
# plt.style.use("fivethirtyeight")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 💾 Import and check out the Data
# 
# Let's import the data


# CELL ********************

USAhousing = pd.read_csv('/lakehouse/default/Files/DMML_Aula5/USA_Housing.csv')
USAhousing.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

USAhousing.info()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

USAhousing.describe()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # 📊 Exploratory Data Analysis (EDA)
# 
# Let's create some simple plots to check out the data!

# CELL ********************

sns.pairplot(USAhousing)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

plt.figure(figsize=(10, 6))  # Set the figure size
plt.scatter(USAhousing['Avg. Area House Age'], USAhousing['Price'], alpha=0.5)  # Create the scatter plot
plt.xlabel('Avg. Area House Age')  # Set the x-axis label
plt.ylabel('Price')  # Set the y-axis label
plt.title('Scatter Plot of Avg. Area House Age vs Price')  # Set the title of the plot
plt.show()  # Display the plot

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************


plt.figure(figsize=(10, 6))  # Set the figure size to match the width parameter
plt.hist(USAhousing['Price'], bins=50, edgecolor='black')  # Create the histogram
plt.xlabel('Price')  # Set the x-axis label
plt.ylabel('Frequency')  # Set the y-axis label
plt.title('Histogram of Price')  # Set the title of the plot
plt.show()  # Display the plot

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

plt.figure(figsize=(10, 6))  # Set the figure size
plt.scatter(USAhousing['Avg. Area Income'], USAhousing['Price'], alpha=0.5)  # Create the scatter plot
plt.xlabel('Avg. Area Income')  # Set the x-axis label
plt.ylabel('Price')  # Set the y-axis label
plt.title('Scatter Plot of Avg. Area Income vs Price')  # Set the title of the plot
plt.show()  # Display the plot

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Select only numeric columns
numeric_columns = USAhousing.select_dtypes(include=['number'])

# Compute the correlation matrix
correlation_matrix = numeric_columns.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))  # Optional: set the figure size
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix Heatmap')
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # 📈 Training a Linear Regression Model
# 
# > Let's now begin to train our regression model! We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable, in this case, the Price column. We will toss out the Address column because it only has text info that the linear regression model can't use.
# 
# ## X and y arrays

# CELL ********************

X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## 🧱 Train Test Split
# 
# Now, let's split the data into a training set and a testing set. We will train our model on the training set and then use the test set to evaluate the model.

# CELL ********************

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# We will define some function to use here. 

# CELL ********************

from sklearn import metrics

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # 📦 Preparing Data For Linear Regression
# > Linear regression is been studied at great length, and there is a lot of literature on how your data must be structured to make the best use of the model.
# 
# > As such, there is a lot of sophistication when talking about these requirements and expectations which can be intimidating. In practice, you can use these rules more as rules of thumb when using Ordinary Least Squares Regression, the most common implementation of linear regression.
# 
# > Try different preparations of your data using these heuristics and see what works best for your problem.
# - **Linear Assumption.** Linear regression assumes that the relationship between your input and output is linear. It does not support anything else. This may be obvious, but it is good to remember when you have a lot of attributes. You may need to transform data to make the relationship linear (e.g. log transform for an exponential relationship).
# - **Remove Noise.** Linear regression assumes that your input and output variables are not noisy. Consider using data cleaning operations that let you better expose and clarify the signal in your data. This is most important for the output variable and you want to remove outliers in the output variable (y) if possible.
# - **Remove Collinearity.** Linear regression will over-fit your data when you have highly correlated input variables. Consider calculating pairwise correlations for your input data and removing the most correlated.
# - **Gaussian Distributions.** Linear regression will make more reliable predictions if your input and output variables have a Gaussian distribution. You may get some benefit using transforms (e.g. log or BoxCox) on your variables to make their distribution more Gaussian looking.
# - **Rescale Inputs:** Linear regression will often make more reliable predictions if you rescale input variables using standardization or normalization.


# CELL ********************

from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_train = scaler.fit_transform(X_train)

# Transform the test data using the fitted scaler
X_test = scaler.transform(X_test)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## ✔️ Linear Regression Model
# 
# We will create the linear regression model using sklearn.linear_model

# CELL ********************

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## ✔️ Model Evaluation
# 
# Let's evaluate the model by checking out it's coefficients and how we can interpret them.

# CELL ********************

# print the intercept
print(lin_reg.intercept_)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])
coeff_df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# > Interpreting the coefficients:
# >- Holding all other features fixed, a 1 unit increase in **Avg. Area Income** is associated with an **increase of \$21.52**.
# >- Holding all other features fixed, a 1 unit increase in **Avg. Area House Age** is associated with an **increase of \$164883.28**.
# >- Holding all other features fixed, a 1 unit increase in **Avg. Area Number of Rooms** is associated with an **increase of \$122368.67**.
# >- Holding all other features fixed, a 1 unit increase in **Avg. Area Number of Bedrooms** is associated with an **increase of \$2233.80**.
# >- Holding all other features fixed, a 1 unit increase in **Area Population** is associated with an **increase of \$15.15**.
# 
# Does this make sense? Probably not because I made up this data.

# MARKDOWN ********************

# ## ✔️ Predictions from our Model
# 
# Let's grab predictions off our test set and see how well it did!

# CELL ********************

pred = lin_reg.predict(X_test)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************


# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, pred, alpha=0.5)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot of True Values vs Predicted Values')
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

from scipy.stats import gaussian_kde

df = pd.DataFrame({'Error Values': (y_test - pred)})

# Create the KDE plot
plt.figure(figsize=(10, 6))
kde = gaussian_kde(df['Error Values'])
x_range = np.linspace(df['Error Values'].min(), df['Error Values'].max(), 1000)
plt.plot(x_range, kde(x_range))
plt.fill_between(x_range, kde(x_range), alpha=0.5)
plt.xlabel('Error Values')
plt.title('KDE Plot of Error Values')
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## ✔️ Regression Evaluation Metrics
# 
# 
# Here are three common evaluation metrics for regression problems:
# 
# - **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:
# $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
# 
# - **Mean Squared Error** (MSE) is the mean of the squared errors:
# $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
# 
# - **Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:
# $$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$
# 
# 📌 Comparing these metrics:
# - **MAE** is the easiest to understand, because it's the average error.
# - **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# - **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.
# 


# CELL ********************

test_pred = lin_reg.predict(X_test)
train_pred = lin_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

# Evaluate the model 
results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, test_pred)]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # ✔️ Polynomial Regression
# 
# 
# > One common pattern within machine learning is to use linear models trained on nonlinear functions of the data. This approach maintains the generally fast performance of linear methods, while allowing them to fit a much wider range of data.
# 
# > For example, a simple linear regression can be extended by constructing polynomial features from the coefficients. In the standard linear regression case, you might have a model that looks like this for two-dimensional data:
# 
# $$\hat{y}(w, x) = w_0 + w_1 x_1 + w_2 x_2$$
# 
# > If we want to fit a paraboloid to the data instead of a plane, we can combine the features in second-order polynomials, so that the model looks like this:
# 
# $$\hat{y}(w, x) = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_1 x_2 + w_4 x_1^2 + w_5 x_2^2$$
# 
# > The (sometimes surprising) observation is that this is still a linear model: to see this, imagine creating a new variable
# 
# $$z = [x_1, x_2, x_1 x_2, x_1^2, x_2^2]$$
# 
# > With this re-labeling of the data, our problem can be written
# 
# $$\hat{y}(w, x) = w_0 + w_1 z_1 + w_2 z_2 + w_3 z_3 + w_4 z_4 + w_5 z_5$$
# 
# > We see that the resulting polynomial regression is in the same class of linear models we’d considered above (i.e. the model is linear in w) and can be solved by the same techniques. By considering linear fits within a higher-dimensional space built with these basis functions, the model has the flexibility to fit a much broader range of data.
# ***


# CELL ********************

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd

# Initialize and fit the PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)

# Transform the training and test sets
X_train_2_d = poly_reg.fit_transform(X_train)
X_test_2_d = poly_reg.transform(X_test)

# Normalize the data
scaler = StandardScaler()
X_train_2_d = scaler.fit_transform(X_train_2_d)
X_test_2_d = scaler.transform(X_test_2_d)

# Initialize and fit the Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_2_d, y_train)

# Predict the test and train sets
test_pred = lin_reg.predict(X_test_2_d)
train_pred = lin_reg.predict(X_train_2_d)

# Print evaluation metrics for the test set
print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')

# Print evaluation metrics for the train set
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

# Evaluate the model without cross-validation
results_df_2 = pd.DataFrame(data=[["Polynomial Regression", *evaluate(y_test, test_pred)]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])

# Concatenate the new results to the existing DataFrame
results_df = pd.concat([results_df, results_df_2], ignore_index=True)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # ✔️ Ridge Regression
# 
# This part of tutorial you should only look at after the next lesson as we have not yet discussed it. 
# 
# > Ridge regression addresses some of the problems of **Ordinary Least Squares** by imposing a penalty on the size of coefficients. The ridge coefficients minimize a penalized residual sum of squares,
# 
# $$\min_{w}\big|\big|Xw-y\big|\big|^2_2+\alpha\big|\big|w\big|\big|^2_2$$
# 
# > $\alpha>=0$ is a complexity parameter that controls the amount of shrinkage: the larger the value of $\alpha$, the greater the amount of shrinkage and thus the coefficients become more robust to collinearity.
# 
# > Ridge regression is an L2 penalized model. Add the squared sum of the weights to the least-squares cost function.
# ***

# CELL ********************

from sklearn.linear_model import Ridge

# Initialize and fit the Ridge regression model
model = Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)
model.fit(X_train, y_train)

# Predict the test and train sets
test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

# Print evaluation metrics for the test set
print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')

# Print evaluation metrics for the train set
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

# Evaluate the model 
results_df_2 = pd.DataFrame(data=[["Ridge Regression", *evaluate(y_test, test_pred)]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])

results_df = pd.concat([results_df, results_df_2], ignore_index=True)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # ✔️ LASSO Regression
# 
# This part of tutorial you should only look at after the next lesson as we have not yet discussed it. 
# 
# > A linear model that estimates sparse coefficients.
# 
# > Mathematically, it consists of a linear model trained with $\ell_1$ prior as regularizer. The objective function to minimize is:
# 
# $$\min_{w}\frac{1}{2n_{samples}} \big|\big|Xw - y\big|\big|_2^2 + \alpha \big|\big|w\big|\big|_1$$
# 
# > The lasso estimate thus solves the minimization of the least-squares penalty with $\alpha \big|\big|w\big|\big|_1$ added, where $\alpha$ is a constant and $\big|\big|w\big|\big|_1$ is the $\ell_1-norm$ of the parameter vector.
# ***

# CELL ********************

from sklearn.linear_model import Lasso
import pandas as pd

# Initialize and fit the Lasso regression model
model = Lasso(alpha=0.1, 
              precompute=True, 
              positive=True, 
              selection='random',
              random_state=42)
model.fit(X_train, y_train)

# Predict the test and train sets
test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

# Print evaluation metrics for the test set
print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')

# Print evaluation metrics for the train set
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

# Evaluate the model 
results_df_2 = pd.DataFrame(data=[["Lasso Regression", *evaluate(y_test, test_pred)]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square'])

results_df = pd.concat([results_df, results_df_2], ignore_index=True)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

## 📊 Models Comparison

results_df.set_index('Model', inplace=True)
results_df['R2 Square'].plot(kind='barh', figsize=(12, 8))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# In this case there is no big difference so better to print out the results

# CELL ********************


results_df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Your turn 
# 
# Now develop your own linear regression in practice
