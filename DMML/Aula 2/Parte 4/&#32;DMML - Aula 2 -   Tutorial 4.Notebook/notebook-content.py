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

# In this tutorial, you'll learn how to create advanced **scatter plots**.
# 
# # Set up the notebook
# 
# As always, we begin by setting up the coding environment.  (_This code is hidden, but you can un-hide it by clicking on the "Code" button immediately below this text, on the right._)

# CELL ********************

import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Load and examine the data
# 
# We'll work with a (_synthetic_) dataset of insurance charges, to see if we can understand why some customers pay more than others.  
# 
# ![tut3_insurance](https://storage.googleapis.com/kaggle-media/learn/images/1nmy2YO.png)
# 
# If you like, you can read more about the dataset [here](https://www.kaggle.com/mirichoi0218/insurance/home).

# CELL ********************

# Path of the file to read
insurance_filepath = "/lakehouse/default/Files/DMML_Aula2/insurance.csv"

# Read the file into a variable insurance_data
insurance_data = pd.read_csv(insurance_filepath)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# As always, we check that the dataset loaded properly by printing the first five rows.

# CELL ********************

insurance_data.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Scatter plots
# 
# To create a simple **scatter plot**, we use the `sns.scatterplot` command and specify the values for:
# - the horizontal x-axis (`x=insurance_data['bmi']`), and 
# - the vertical y-axis (`y=insurance_data['charges']`).

# CELL ********************

sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# The scatterplot above suggests that [body mass index](https://en.wikipedia.org/wiki/Body_mass_index) (BMI) and insurance charges are **positively correlated**, where customers with higher BMI typically also tend to pay more in insurance costs.  (_This pattern makes sense, since high BMI is typically associated with higher risk of chronic disease._)
# 
# To double-check the strength of this relationship, you might like to add a **regression line**, or the line that best fits the data.  We do this by changing the command to `sns.regplot`.

# CELL ********************

sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# # Color-coded scatter plots
# 
# We can use scatter plots to display the relationships between (_not two, but..._) three variables!  One way of doing this is by color-coding the points.  
# 
# For instance, to understand how smoking affects the relationship between BMI and insurance costs, we can color-code the points by `'smoker'`, and plot the other two columns (`'bmi'`, `'charges'`) on the axes.

# CELL ********************

sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# This scatter plot shows that while nonsmokers to tend to pay slightly more with increasing BMI, smokers pay MUCH more.
# 
# To further emphasize this fact, we can use the `sns.lmplot` command to add two regression lines, corresponding to smokers and nonsmokers.  (_You'll notice that the regression line for smokers has a much steeper slope, relative to the line for nonsmokers!_)

# CELL ********************

sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# The `sns.lmplot` command above works slightly differently than the commands you have learned about so far:
# - Instead of setting `x=insurance_data['bmi']` to select the `'bmi'` column in `insurance_data`, we set `x="bmi"` to specify the name of the column only.  
# - Similarly, `y="charges"` and `hue="smoker"` also contain the names of columns.  
# - We specify the dataset with `data=insurance_data`.
# 
# Finally, there's one more plot that you'll learn about, that might look slightly different from how you're used to seeing scatter plots.  Usually, we use scatter plots to highlight the relationship between two continuous variables (like `"bmi"` and `"charges"`).  However, we can adapt the design of the scatter plot to feature a categorical variable (like `"smoker"`) on one of the main axes.  We'll refer to this plot type as a **categorical scatter plot**, and we build it with the `sns.swarmplot` command.

# CELL ********************

sns.swarmplot(x=insurance_data['smoker'],
              y=insurance_data['charges'])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Among other things, this plot shows us that:
# - on average, non-smokers are charged less than smokers, and
# - the customers who pay the most are smokers; whereas the customers who pay the least are non-smokers.
# 
# # What's next?
# 
# Apply your new skills to solve a real-world scenario with a **[coding exercise]** notebook!
