# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "jupyter",
# META     "jupyter_kernel_name": "python3.11"
# META   }
# META }

# MARKDOWN ********************

# # Introduction 
# 
# ![Decision_Tree_Header](https://raw.githubusercontent.com/satishgunjal/images/master/Decision_Tree_Header.png)
# 
# Decision tree algorithm belongs to the family of supervised learning algorithms. Unlike other supervised learning algorithms decision tree can be used to solve regression and classification problems. The goal of decision tree is to create training model that can predict class(single or multi) or value by learning simple decision rules from training data.
# Decision tree form a flow chart like structure that's why they are very easy to interpret and understand. It is one of the few ML algorithm where its very easy to visualize and analyze the internal working of algorithm.
# 
# Just like flowchart, decision tree contains different types of nodes and branches. Every decision node represent the test on feature and based on the test result it will either form another branch or the leaf node. Every branch represents the decision rule and leaf node represent the final outcome.
# 
# ![decision tree](https://raw.githubusercontent.com/satishgunjal/images/master/Decision_Tree.png)
# 
# Types of decision tree
# * Classification decision trees − In this kind of decision trees, the decision variable is categorical. 
# * Regression decision trees − In this kind of decision trees, the decision variable is continuous


# MARKDOWN ********************

# # Inner Workings Of Decision Tree
# * At the root node decision tree selects feature to split the data in two major categories.
# * So at the end of root node we have two decision rules and two sub trees
# * Data will again be divided in two categories in each sub tree
# * This process will continue until every training example is grouped together.
# * So at the end of decision tree we end up with leaf node. Which represent the class or a continuous value that we are trying predict
# 
# ## Criteria To Split The Data
# The objective of decision tree is to split the data in such a way that at the end we have different groups of data which has more similarity and less randomness/impurity. In order to achieve this, every split in decision tree must reduce the randomness.
# Decision tree uses 'entropy' or 'gini' selection criteria to split the data.
# Note: We are going to use sklearn library to test classification and regression. 'entropy' or 'gini' are selection criteria for classifier whereas “mse”, “friedman_mse” and “mae” are selection criteria for regressor.
# 
# ### Entropy
# In order to find the best feature which will reduce the randomness after a split, we can compare the randomness before and after the split for every feature. In the end we choose the feature which will provide the highest reduction in randomness. Formally randomness in data is known as 'Entropy' and difference between the 'Entropy' before and after split is known as 'Information Gain'. Since in case of decision tree we may have multiple branches, information gain formula can be written as,
# 
# ```
#     Information Gain= Entropy(Parent Decision Node)–(Average Entropy(Child Nodes))
# ```
# 
# 'i' in below Entropy formula represent the target classes 
# 
#    ![entropy_formula](https://raw.githubusercontent.com/satishgunjal/images/master/entropy_formula.png)
# 
# So in case of 'Entropy', decision tree will split the data using the feature that provides the highest information gain.
# 
# ### Gini
# 
# We have not look to Gini in the classes but it is also an option, here is how Gini works.
# In case of gini impurity, we pick a random data point in our dataset. Then randomly classify it according to the class distribution in the dataset. So it becomes very important to know the accuracy of this random classification. Gini impurity gives us the probability of incorrect classification. We’ll determine the quality of the split by weighting the impurity of each branch by how many elements it has. Resulting value is called as 'Gini Gain' or 'Gini Index'. This is what’s used to pick the best split in a decision tree. Higher the Gini Gain, better the split
# 
# 'i' in below Gini formula represent the target classes 
# 
#    ![gini_formula](https://raw.githubusercontent.com/satishgunjal/images/master/gini_formula.png)
# 
# So in case of 'gini', decision tree will split the data using the feature that provides the highest gini gain.
# 


# MARKDOWN ********************

# # Advantages Of Decision Tree
# * Simple to understand and to interpret. Trees can be visualized.
# * Requires little data preparation. Other techniques often require data normalization, dummy variables need to be created and blank values to be removed. Note however that this module does not support missing values.
# * Able to handle both numerical and categorical data.
# * Able to handle multi-output problems.
# * Uses a white box model. Results are easy to interpret.
# * Possible to validate a model using statistical tests. That makes it possible to account for the reliability of the model.

# MARKDOWN ********************

# # Disadvantages Of Decision Tree
# * Decision-tree learners can create over-complex trees that do not generalize the data well. This is called overfitting. Mechanisms such as pruning, setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are necessary to avoid this problem.
# * Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This problem is mitigated by using decision trees within an ensemble.
# * Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.

# MARKDOWN ********************

# # Classification Problem Example
# For classification exercise we are going to use sklearns iris plant dataset.
# Objective is to classify iris flowers among three species (setosa, versicolor or virginica) from measurements of length and width of sepals and petals
# 
# ## Understanding the IRIS dataset
# * iris.DESCR > Complete description of dataset
# * iris.data > Data to learn. Each training set is 4 digit array of features. Total 150 training sets
# * iris.feature_names > Array of all 4 feature ['sepal length (cm)','sepal width cm)','petal length (cm)','petal width (cm)']
# * iris.filename > CSV file name
# * iris.target > The classification label. For every training set there is one classification label(0,1,2). Here 0 for setosa, 1 for versicolor and 2 for virginica
# * iris.target_names > the meaning of the features. It's an array >> ['setosa', 'versicolor', 'virginica']
# 
# From above details its clear that X = 'iris.data' and y= 'iris.target'
# 
# ![Iris_setosa](https://raw.githubusercontent.com/satishgunjal/images/master/iris_species.png)
# 


# MARKDOWN ********************

# ## Import Libraries
# * pandas: Used for data manipulation and analysis
# * numpy : Numpy is the core library for scientific computing in Python. It is used for working with arrays and matrices.
# * datasets: Here we are going to use ‘iris’ and 'boston house prices' dataset
# * model_selection: Here we are going to use model_selection.train_test_split() for splitting the data
# * tree: Here we are going to decision tree classifier and regressor
# * graphviz: Is used to export the tree into Graphviz format using the export_graphviz exporter

# CELL ********************

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn import tree
import graphviz
import mlflow
mlflow.autolog(disable=True)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Load The Data

# CELL ********************

iris = datasets.load_iris()
print('Dataset structure= ', dir(iris))

df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['target'] = iris.target
df['flower_species'] = df.target.apply(lambda x : iris.target_names[x]) # Each value from 'target' is used as index to get corresponding value from 'target_names' 

print('Unique target values=',df['target'].unique())

df.sample(5)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Note that, target value 0 = setosa, 1 = versicolor and 2 = virginica
# 
# Let visualize the feature values for each type of flower

# CELL ********************

# label = 0 (setosa)
df[df.target == 0].head(3)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# label = 1 (versicolor)
df[df.target == 1].head(3)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# label = 2 (verginica)
df[df.target == 2].head(3)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Build Machine Learning Model

# CELL ********************

#Lets create feature matrix X  and y labels
X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = df[['target']]

print('X shape=', X.shape)
print('y shape=', y.shape)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### Create Test And Train Dataset
# * We will split the dataset, so that we can use one set of data for training the model and one set of data for testing the model
# * We will keep 20% of data for testing and 80% of data for training the model
# * If you want to learn more about it, please refer [Train Test Split tutorial](https://satishgunjal.com/train_test_split/)

# CELL ********************

X_train,X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.2, random_state= 1)
print('X_train dimension= ', X_train.shape)
print('X_test dimension= ', X_test.shape)
print('y_train dimension= ', y_train.shape)
print('y_train dimension= ', y_test.shape)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Now lets train the model using Decision Tree

# CELL ********************

"""
To obtain a deterministic behaviour during fitting always set value for 'random_state' attribute
Also note that default value of criteria to split the data is 'gini'
"""
cls = tree.DecisionTreeClassifier(random_state= 1)
cls.fit(X_train ,y_train)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### Testing The Model
# * For testing we are going to use the test data only
# * Question: Predict the species of 10th, 20th and 29th test example from test data

# CELL ********************

print('Actual value of species for 10th training example=',iris.target_names[y_test.iloc[10]][0])
print('Predicted value of species for 10th training example=', iris.target_names[cls.predict([X_test.iloc[10]])][0])

print('\nActual value of species for 20th training example=',iris.target_names[y_test.iloc[20]][0])
print('Predicted value of species for 20th training example=', iris.target_names[cls.predict([X_test.iloc[20]])][0])

print('\nActual value of species for 30th training example=',iris.target_names[y_test.iloc[29]][0])
print('Predicted value of species for 30th training example=', iris.target_names[cls.predict([X_test.iloc[29]])][0])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### Model Score
# Check the model score using test data

# CELL ********************

cls.score(X_test, y_test)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Visualize The Decision Tree
# We will use plot_tree() function from sklearn to plot the tree and then export the tree in Graphviz format using the export_graphviz exporter. Results will be saved in iris_decision_tree.pdf file

# CELL ********************

tree.plot_tree(cls) 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

dot_data = tree.export_graphviz(cls, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris_decision_tree") 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

dot_data = tree.export_graphviz(cls, out_file=None, 
                      feature_names=iris.feature_names,  
                      class_names=iris.target_names,  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Regression Problem Example
# For regression exercise we are going to use sklearns Boston house prices dataset
# Objective is to predict house price based on available data
# 
# ## Understanding the Boston house dataset
# * boston.DESCR > Complete description of dataset
# * boston.data > Data to learn. There are 13 features, Median Value (attribute 14) is usually the target. Total 506 training sets
#     - CRIM     per capita crime rate by town
#     - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#     - INDUS    proportion of non-retail business acres per town
#     - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#     - NOX      nitric oxides concentration (parts per 10 million)
#     - RM       average number of rooms per dwelling
#     - AGE      proportion of owner-occupied units built prior to 1940
#     - DIS      weighted distances to five Boston employment centres
#     - RAD      index of accessibility to radial highways
#     - TAX      full-value property-tax rate per USD 10,000
#     - PTRATIO  pupil-teacher ratio by town
#     - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#     - LSTAT    % lower status of the population
#     - MEDV     Median value of owner-occupied homes in USD 1000's
# * boston.feature_names > Array of all 13 features ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']
# * boston.filename > CSV file name
# * boston.target > The price valueis in $1000’s
# 
# From above details its clear that X = 'boston.data' and y= 'boston.target'


# CELL ********************

import pandas as pd
import numpy as np

df = pd.read_csv(
    filepath_or_buffer="http://lib.stat.cmu.edu/datasets/boston",
    delim_whitespace=True,
    skiprows=21,
    header=None,
)

columns = [
    'CRIM',
    'ZN',
    'INDUS',
    'CHAS',
    'NOX',
    'RM',
    'AGE',
    'DIS',
    'RAD',
    'TAX',
    'PTRATIO',
    'B',
    'LSTAT',
    'MEDV',
]

features_name = [
    'CRIM',
    'ZN',
    'INDUS',
    'CHAS',
    'NOX',
    'RM',
    'AGE',
    'DIS',
    'RAD',
    'TAX',
    'PTRATIO',
    'B',
    'LSTAT',
]

#Flatten all the values into a single long list and remove the nulls
values_w_nulls = df.values.flatten()
all_values = values_w_nulls[~np.isnan(values_w_nulls)]



#Reshape the values to have 14 columns and make a new df out of them
df = pd.DataFrame(
    data = all_values.reshape(-1, len(columns)),
    columns = columns,
)

#rename MDEV to target
df.rename(columns={'MEDV': 'target'}, inplace=True)

df

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Build Machine Learning Model

# CELL ********************

#Lets create feature matrix X  and y labels
X = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
y = df[['target']]

print('X shape=', X.shape)
print('y shape=', y.shape)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### Create Test And Train Dataset
# * We will split the dataset, so that we can use one set of data for training the model and one set of data for testing the model
# * We will keep 20% of data for testing and 80% of data for training the model


# CELL ********************

X_train,X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.2, random_state= 1)
print('X_train dimension= ', X_train.shape)
print('X_test dimension= ', X_test.shape)
print('y_train dimension= ', y_train.shape)
print('y_train dimension= ', y_test.shape)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Now lets train the model using Decision Tree

# CELL ********************

"""
To obtain a deterministic behaviour during fitting always set value for 'random_state' attribute
To keep the tree simple I am using max_depth = 3
Also note that default value of criteria to split the data is 'mse' (mean squared error)
mse is equal to variance reduction as feature selection criterion and minimizes the L2 loss using the mean of each terminal node
"""
dtr = tree.DecisionTreeRegressor(max_depth= 3,random_state= 1)
dtr.fit(X_train ,y_train)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### Testing The Model
# * For testing we are going to use the test data only
# * Question: predict the values for every test set in test data

# CELL ********************

predicted_price= pd.DataFrame(dtr.predict(X_test), columns=['Predicted Price'])
actual_price = pd.DataFrame(y_test, columns=['target'])
actual_price = actual_price.reset_index(drop=True) # Drop the index so that we can concat it, to create new dataframe
df_actual_vs_predicted = pd.concat([actual_price,predicted_price],axis =1)
df_actual_vs_predicted.T

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ### Model Score
# Check the model score using test data

# CELL ********************

dtr.score(X_test, y_test)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Visualize The Decision Tree
# We will use plot_tree() function from sklearn to plot the tree and then export the tree in Graphviz format using the export_graphviz exporter. Results will be saved in boston_decision_tree.pdf file

# CELL ********************

tree.plot_tree(dtr) 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

dot_data = tree.export_graphviz(dtr, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("boston_decision_tree") 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

dot_data = tree.export_graphviz(dtr, out_file=None, 
                      feature_names=features_name,  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }
