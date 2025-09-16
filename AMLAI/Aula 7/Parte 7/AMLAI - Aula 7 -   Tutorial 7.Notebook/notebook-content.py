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

# # Time Series Forecasting with ARIMA Models 
# 
# We have already explored a lot around time series, now you will learn more on time series realted with ARIMA. Understanding how to work with time series and how to apply analytical and forecasting techniques are critical for every aspiring data scientist. 
# We will go through the basic techniques to work with time-series data, starting from data manipulation, analysis, and visualization to understand your data and prepare it for and then using the statistical, machine, and deep learning techniques for forecasting and classification. 
# 
# 


# MARKDOWN ********************

# # 1. ARMA Models
# We will start with a small introduction to stationarity and how this is important for ARMA models. Then we will revise how to test for stationarity by eye and with a standard statistical test. If you would like to get more information about these topics, you can check my previous article Time Series Analysis In Python as they are covered in more detail in it. Finally, you’ll learn the basic structure of ARMA models and use this to generate some ARMA data and fit an ARMA model.
# 
# We will use the candy production dataset, which represents the monthly candy production in the US between 1972 and 2018. Specifically, we will be using the industrial production index IPG3113N. This is the total amount of sugar and confectionery products produced in the USA per month, as a percentage of the January 2012 production. So 120 would be 120% of the January 2012 industrial production.

# MARKDOWN ********************

# ## 1.1. Introduction to stationarity
# Stationary means that the distribution of the data doesn’t change with time. For a time series to be stationary it must fulfill three criteria:
# 
# The series has zero trends. It isn’t growing or shrinking.
# The variance is constant. The average distance of the data points from the zero line isn’t changing.
# The autocorrelation is constant. How each value in the time series is related to its neighbors stays the same.
# The importance of stationarity comes from that to model a time series, it must be stationary. The reason for this is that modeling is all about estimating parameters that represent the data, therefore if the parameters of the data are changing with time, it will be difficult to estimate all the parameters.
# 
# Let’s first load and plot the monthly candy production dataset:

# CELL ********************

import pandas as pd
import matplotlib.pyplot as plt
import mlflow
mlflow.autolog(disable=True)
# Load in the time series
candy = pd.read_csv('/lakehouse/default/Files/AMLAI_Aula7/candy_production.csv', 
            index_col='observation_date',
            parse_dates=True)
# change the plot style into fivethirtyeight 
plt.style.use('fivethirtyeight')

# Plot and show the time series on axis ax1
fig, ax1 = plt.subplots()
candy.plot(ax=ax1, figsize=(12,10))
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Generally, in machine learning, you have a training set on which you fit your model, and a test set, on which you will test your predictions against. Time series forecasting is just the same. Our train-test split will be different. We use the past values to make future predictions, and so we will need to split the data in time. We train on the data earlier in the time series and test on the data that comes later. We can split time series at a given date as shown below using the DataFrame’s **.loc** method.

# CELL ********************

# Split the data into a train and test set
candy_train = candy.loc[:'2006']
candy_test = candy.loc['2007':]

# Create an axis
fig, ax = plt.subplots()

# Plot the train and test sets on the axis ax
candy_train.plot(ax=ax, figsize=(12,10))
candy_test.plot(ax=ax)
plt.title('train - test split of the monthly production of candy in US')
plt.xlabel('Date')
plt.ylabel('Production')
plt.show()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 1.2. Making a time series stationary
# 
# There are many ways to test stationary, one of them with eyes, and others are more formal using statistical tests. There are also ways to transform non-stationary time series into stationary ones. We’ll address both of these in this subsection and then you’ll be ready to start modeling.
# 
# The most common test for identifying whether a time series is non-stationary is the augmented Dicky-Fuller test. This is a statistical test, where the null hypothesis is that your time series is non-stationary due to trends. We can implement the augmented Dicky-Fuller test using statsmodels. First, we import the adfuller function as shown, then we can run it on the candy production time series.

# CELL ********************

from statsmodels.tsa.stattools import adfuller
results = adfuller(candy)
print(results)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# The results object is a tuple. The zeroth element is the test statistic, in this case, it is -1.77. The more negative this number is, the more likely that the data is stationary. The next item in the results tuple is the test p-value. Here it’s 0.3. If the p-value is smaller than 0.05, we reject the null hypothesis and assume our time series must be stationary. The last item in the tuple is a dictionary. This stores the critical values of the test statistic which equate to different p-values. In this case, if we wanted a p-value of 0.05 or below, our test statistic needed to be below -2.86.
# 
# Based on this result, we are sure that the time series is non-stationary. Therefore, we will need to transform the data into a stationary form before we can model it. We can think of this a bit like feature engineering in classic machine learning. One very common way to make a time series stationary is to take its difference. This is where from each value in our time series we subtract the previous value.

# CELL ********************

# Calculate the first difference and drop the nans
candy_diff = candy.diff()
candy_diff = candy_diff.dropna()

# Run test and print
result_diff = adfuller(candy_diff)
print(result_diff)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# From the results, we can see that now the time series are stationary. This time, taking the difference was enough to make it stationary, but for other time series, we may need to make the difference more than once or do other transformations. Sometimes we will need to perform other transformations to make the time series stationary. This could be to take the log, or the square root of a time series, or to calculate the proportional change. It can be hard to decide which of these to do, but often the simplest solution is the best one.

# MARKDOWN ********************

# ## 1.3. Introduction to AR, MA and ARMA models
# 
# In an autoregressive (AR) model, we regress the values of the time series against previous values of this same time series. The equation for a simple AR model is shown below:
# 
# y(t) = a(1) * y(t-1) + ϵ(t)
# 
# The value of the time series at the time (t) is the value of the time series at the previous step multiplied with parameter a(1) added to a noise or shock term ϵ(t). The shock term is white noise, meaning each shock is random and not related to the other shocks in the series. a(1) is the autoregressive coefficient at lag one. Compare this to a simple linear regression where the dependent variable is y(t) and the independent variable is y(t-1). The coefficient a(1) is just the slope of the line and the shocks are the residuals of the line.
# 
# This is a first-order AR model. The order of the model is the number of time lags used. An order two AR model has two autoregressive coefficients and has two independent variables, the series at lag one and the series at lag two. More generally, we use p to mean the order of the AR model. This means we have p autoregressive coefficients and use p lags.
# 
# In a moving average (MA) model, we regress the values of the time series against the previous shock values of this same time series. The equation for a simple MA model is shown below:
# 
# y(t) = m(1)*ϵ(t-1) + ϵ(t)
# 
# The value of the time series y(t)is m(1) times the value of the shock at the previous step; plus a shocking term for the current time step. This is a first-order MA model. Again, the order of the model means how many time lags we use. An MA two model would include shocks from one and two steps ago. More generally, we use q to mean the order of the MA model.
# 
# An ARMA model is a combination of the AR and MA models. The time series is regressed on the previous values and the previous shock terms. This is an ARMA-one-one model. More generally we use ARMA(p,q) to define an ARMA model. The p tells us the order of the autoregressive part of the model and the q tells us the order of the moving average part.
# 
# y(t) = a (1)*y(t-1) + m(1)* ϵ(t-1) + ϵ(t)
# 
# Using the statsmodels package, we can both fit ARMA models and create ARMA data. Let’s take this ARMA-one-one model. Say we want to simulate data with these coefficients. First, we import the arma-generate-sample function. Then we make lists for the AR and MA coefficients. Note that both coefficient lists start with one. This is for the zero-lag term and we will always set this to one. We set the lag one AR coefficient as 0.5 and the MA coefficient as 0.2. We generate the data, passing in the coefficients, the number of data points to create, and the standard deviation of the shocks. Here, we actually pass in the negative of the AR coefficients we desire. This is a quirk we will need to remember.

# CELL ********************

from statsmodels.tsa.arima_process import arma_generate_sample
ar_coefs = [1, -0.5] 
ma_coefs = [1, 0.2]
y = arma_generate_sample(ar_coefs, ma_coefs, nsample=100, scale=0.5)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# The generated data can be represented with this equation:
# 
# y(t) = 0.5y(t−1) + 0.2* ϵ(t−1) + ϵ(t).
# 
# ![image.png](attachment:660d3c6c-d83f-41b5-8b19-be385ad9f278.png)
# 
# Fitting is covered in the next section, but here is a quick peek at how we might fit this data. First, we import the ARMA model class. We instantiate the model, feed it the data and define the model order. Then finally we fit.

# ATTACHMENTS ********************

# ATTA {
# ATTA   "660d3c6c-d83f-41b5-8b19-be385ad9f278.png": {
# ATTA     "image/png": "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"
# ATTA   }
# ATTA }

# CELL ********************

from statsmodels.tsa.arima.model import ARIMA
# Instantiate model object
model = ARIMA(y, order=(1,0,1))
# Fit model
results = model.fit()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # 2. Fitting the Future
# In this section, you’ll learn how to use the elegant statsmodels package to fit ARMA, ARIMA, and ARMAX models. Then you’ll use your models to predict the uncertain future of Amazon stock prices.
# 
# ## 2.1. Fitting time series models
# 
# We had a quick look at fitting time series models in the last section but let’s have a closer look. To fit these models we first import the ARIMA model class from the statsmodels package. We create a model object and define the model order, we must also feed in the training data. The data can be a pandas dataframe, a pandas series, or a NumPy array. Remember that the order for an ARIMA model is (p,d,q) p is the autoregressive lags, d is the order of the difference, and q is the moving average lags. d is always an integer, while p and q may either be integers or lists of integers. To fit an AR model we can simply use the ARMA class with q equal to zero. To fit an MA model, we set p equal to zero.
# 
# Let’s have a look at the result summary of the fitted model :

# CELL ********************

from statsmodels.tsa.arima.model import ARIMA
# Instantiate model object
model = ARIMA(y, order=(1,0,1))
# Fit model
results = model.fit()
print(results.summary())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# The top section includes useful information such as the order of the model that we fit, the number of observations or data points, and the name of the time series. The S.D. of innovations is the standard deviation of the shock terms.
# 
# The next section of the summary shows the fitted model parameters. Here we fitted an ARIMA(1,0,1) model, so the model has AR-lag-1 and lag-1 coefficients. In the table, these are the ar.L1 and ma.L1 rows. The lag-1 MA coefficient is in the last row. The first column shows the model coefficients whilst the second column shows the standard error in these coefficients. This is the uncertainty on the fitted coefficient values.
# 
# One possible extension to the ARMA model is to use exogenous inputs to create the ARMAX model. This means that we model the time series using other independent variables as well as the time series itself. This is like a combination between an ARMA model and a normal linear regression model. The equations for two simple ARMA and ARMAX models are shown here. The only difference is one extra term. We add a new independent variable z(t) multiplied by its coefficient x(1). Let’s think of an example where ARMAX might be useful.
# 
# ARMA(1,1) model :
# 
# y(t) = a(1) y(t-1) + m(1) ϵ(t-1) + ϵ(t)
# 
# ARMAX(1,1) model :
# 
# y = x(1)* z(t) + a(1) y(t-1) + m (1)ϵ(t-1) + ϵ(t)
# 
# We can fit an ARMAX model using the same ARMA model class we used before. The only difference is that we will now feed in our exogenous variable using the exog keyword. The model order and the fitting procedure are just the same.
# 
# ## 2.2. Forecasting
# After introducing how to fit ARIMA models to data, let’s see how to use them to forecast and predict the future. let’s take an example of time series represented by an AR(1) model. At any time point in the time series, we can predict the next values by multiplying the previous value with the lag-one AR coefficient. If the previous value was 15 and coefficient a-one is 0.5, we would estimate the next value is 7.5. If the shock term had a standard deviation of 1, we would predict our lower and upper uncertainty limits to be 6.5 and 8.5. This type of prediction is called one-step-ahead prediction. Below is its equation:
# 
# y = 0.5 x 15 + ϵ(t)
# 
# ## 2.3. ARIMA models for non-stationary time series
# If the time series you are trying to forecast is non-stationary, you will not be able to apply the ARMA model to it. We first have to make the difference to make it stationary and then we can use the ARMA model for it. However, when we do this, we will have a model which is trained to predict the value of the difference of the time series. What we really want to predict is not the difference, but the actual value of the time series. We can achieve this by carefully transforming our prediction of the differences.
# 
# We start with predictions of the difference values The opposite of taking the difference is taking the cumulative sum or integral. We will need to use this transform to go from predictions of the difference values to predictions of the absolute values.
# 
# We can do this using the np.cumsum function. If we apply this function we now have a prediction of how much the time series changed from its initial value over the forecast period. To get an absolute value we need to add the last value of the original time series to this.

# CELL ********************

from numpy import cumsum
mean_forecast = cumsum(candy_diff) + candy.iloc[-1, 0]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# These steps of starting with non-stationary data; differencing to make it stationary; and then integrating the forecast are very common in time series modeling. This is a lot of work! But thankfully, there is an extension of the ARMA model which does it for us! This is the autoregressive integrated moving average model (ARIMA).
# 
# We can implement an ARIMA model using the SARIMAX model class from statsmodels. The ARIMA model has three model orders. These are p the autoregressive order; d the order of differencing, and q the moving average order. In the previous section, we were setting the middle order parameter d to zero. If d is zero we simply have an ARMA model.
# 
# When we use this model, we pass it in a non-differenced time series and the model order. When we want to difference the time series data just once and then apply an ARMA(2,1) model. This is achieved by using an ARIMA(2,1,1) model. After we have stated the difference parameter we don’t need to worry about differencing anymore. We fit the model as before and make forecasts. The differencing and integration steps are all taken care of by the model object. This is a much easier way to get a forecast for non-stationary time series!

# MARKDOWN ********************

# We must still be careful about selecting the right amount of differencing. Remember, we difference our data only until it is stationary and no more. We will work this out before we apply our model, using the augmented Dicky-Fuller test to decide the difference order. So by the time we come to apply a model we already know the degree of differencing we should apply.
# 
# Let's apply this to real data. The data that will be used is the **amazon stock price** data. We will apply the two methods. First using ARMA models on the data with a difference and using ARIMA model with built-in difference.
# First, the data will be uploaded and plotted.

# CELL ********************

amazon = pd.read_csv('/lakehouse/default/Files/AMLAI_Aula7/candy_production.csv',
                     index_col='observation_date',
                     parse_dates=True)
amazon.plot()
plt.title('Amazon stock price change with time')
plt.ylabel('Stock price')

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# First, we will apply the Adfuller-Dickey test to know whether the time series is stationary or not.

# CELL ********************

from statsmodels.tsa.stattools import adfuller

# Run Dicky-Fuller test
result = adfuller(amazon)

# Print test statistic
print('The test stastics:', result[0])

# Print p-value
print("The p-value:",result[1])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# The p-value is bigger than 0.05, therefore we cannot reject the null hypothesis and the time series is considered to be non-stationary. Therefore we will take the first difference and check whether this will make it stationary or not using also Adfuller-Dickey test.
# 


# CELL ********************

# take the first diff
amazon_diff = amazon.diff()
amazon_diff.dropna(inplace=True)

# Run Dicky-Fuller test
result = adfuller(amazon_diff)

# Print test statistic
print('The test stastics:', result[0])

# Print p-value
print("The p-value:",result[1])


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# The p-value after taking the first difference of the amazon stock price time series is less than 0.05, so we can reject the null hypothesis and the data now is considered stationary. For the modeling step, we can follow one of the two paths mentioned above. First, use the ARMA model and apply it to the data with the first difference. Then the **np.cumsum** function will be used for the prediction of the actual data not the data with the difference.

# CELL ********************

from statsmodels.tsa.arima.model import ARIMA
# Instantiate model object
model = ARIMA(amazon_diff, order=(1,0,1))
# Fit model
results = model.fit()
print(results.summary())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# The second method is to use the ARIMA model and use the actual data and use the difference parameter in the ARIMA function.

# CELL ********************

from statsmodels.tsa.arima.model import ARIMA
# Instantiate model object
model = ARIMA(amazon, order=(1,1,1))
# Fit model
results = model.fit()
print(results.summary())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## How to find the order of differencing (d) in ARIMA model
# 
# 
# 
# - As stated earlier, the purpose of differencing is to make the time series stationary. But we should be careful to not over-difference the series. An over differenced series may still be stationary, which in turn will affect the model parameters.
# 
# 
# - So we should determine the right order of differencing. The right order of differencing is the minimum differencing required to get a near-stationary series which roams around a defined mean and the ACF plot reaches to zero fairly quick.
# 
# 
# - If the autocorrelations are positive for many number of lags (10 or more), then the series needs further differencing. On the other hand, if the lag 1 autocorrelation itself is too negative, then the series is probably over-differenced.
# 
# 
# - If we can’t really decide between two orders of differencing, then we go with the order that gives the least standard deviation in the differenced series.
# 
# 
# - Now, we will explain these concepts with the help of an example as follows:-
# 


# MARKDOWN ********************

# - First, I will check if the series is stationary using the **Augmented Dickey Fuller test (ADF Test)**, from the statsmodels package. The reason being is that we need differencing only if the series is non-stationary. Else, no differencing is needed, that is, d=0.
# 
# 
# - The null hypothesis (Ho) of the ADF test is that the time series is non-stationary. So, if the p-value of the test is less than the significance level (0.05) then we reject the null hypothesis and infer that the time series is indeed stationary.
# 
# 
# - So, in our case, if P Value > 0.05 we go ahead with finding the order of differencing.
# 
# 
# ### Import data

# CELL ********************

path = '/lakehouse/default/Files/AMLAI_Aula7/dataset.txt'

df = pd.read_csv(path)

df.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(df.value.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# - Since p-value(1.00) is greater than the significance level(0.05), let’s difference the series and see how the autocorrelation plot looks like.

# CELL ********************

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})


# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.value); axes[0, 0].set_title('Original Series')
plot_acf(df.value, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df.value.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.value.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df.value.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1])

plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# - For the above data, we can see that the time series reaches stationarity with two orders of differencing.


# MARKDOWN ********************

# ## How to find the order of the AR term (p)
# 
# 
# 
# - The next step is to identify if the model needs any AR terms. We will find out the required number of AR terms by inspecting the **Partial Autocorrelation (PACF) plot**.
# 
# 
# - **Partial autocorrelation** can be imagined as the correlation between the series and its lag, after excluding the contributions from the intermediate lags. So, PACF sort of conveys the pure correlation between a lag and the series. This way, we will know if that lag is needed in the AR term or not.
# 
# 
# - Partial autocorrelation of lag (k) of a series is the coefficient of that lag in the autoregression equation of Y.
# 
# 
# $$Yt = \alpha0 + \alpha1 Y{t-1} + \alpha2 Y{t-2} + \alpha3 Y{t-3}$$
# 
# 
# - That is, suppose, if Y_t is the current series and Y_t-1 is the lag 1 of Y, then the partial autocorrelation of lag 3 (Y_t-3) is the coefficient $\alpha_3$ of Y_t-3 in the above equation.
# 
# 
# - Now, we should find the number of AR terms. Any autocorrelation in a stationarized series can be rectified by adding enough AR terms. So, we initially take the order of AR term to be equal to as many lags that crosses the significance limit in the PACF plot.
# 


# CELL ********************

# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.value.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(df.value.diff().dropna(), ax=axes[1])

plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# - We can see that the PACF lag 1 is quite significant since it is well above the significance line. So, we will fix the value of p as 1.

# MARKDOWN ********************

# ## How to find the order of the MA term (q)
# 
# 
# 
# - Just like how we looked at the PACF plot for the number of AR terms, we will look at the ACF plot for the number of MA terms. An MA term is technically, the error of the lagged forecast.
# 
# 
# - The ACF tells how many MA terms are required to remove any autocorrelation in the stationarized series.
# 
# 
# - Let’s see the autocorrelation plot of the differenced series.

# CELL ********************

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.value.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(df.value.diff().dropna(), ax=axes[1])

plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# - We can see that couple of lags are well above the significance line. So, we will fix q as 2. If there is any doubt, we will go with the simpler model that sufficiently explains the Y.

# MARKDOWN ********************

# 
# ## Your turn 
# 
# Move to the exercise notebook where you will apply what you have learn. 
# 
# 

