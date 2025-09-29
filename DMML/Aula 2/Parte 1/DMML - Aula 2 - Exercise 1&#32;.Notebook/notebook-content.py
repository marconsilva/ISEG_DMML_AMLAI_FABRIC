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
# META     },
# META     "environment": {
# META       "environmentId": "be6440c5-d6f3-4252-85b9-2cfd4b258ea1",
# META       "workspaceId": "03f3982f-785f-4a2f-8ec0-4be54060ee7b"
# META     }
# META   }
# META }

# MARKDOWN ********************

# In this exercise, you will write your first lines of code !
# 
# ## Setup
# 
# First, you'll learn how to run code, and we'll start with the code cell below.  (Remember that a **code cell** in a notebook is just a gray box containing code that we'd like to run.)
# - Begin by clicking inside the code cell.  
# - Click on the blue triangle (in the shape of a "Play button") that appears to the left of the code cell.
# - If your code was run sucessfully, you will see `Setup Complete` as output below the cell.
# 
# ![ex0_run_code](https://storage.googleapis.com/kaggle-media/learn/images/4NzqJ7G.png)

# MARKDOWN ********************

# The code cell below imports and configures the Python libraries that you need to complete the exercise.
# 
# Click on the cell and run it.

# CELL ********************

%pip install /lakehouse/default/Files/Env/learntools-0.3.4-py2.py3-none-any.whl
%pip install seaborn

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Set up code checking
import os
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex1 import *
print("Setup Complete")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# The code you just ran sets up the system to give you feedback on your work.  You'll learn more about the feedback system in the next step.
# 
# ## Step 1: Explore the feedback system
# 
# Each  exercise lets you test your new skills with a real-world dataset.  Along the way, you'll receive feedback on your work.  You'll see if your answer is right, get customized hints, and see the official solution (_if you'd like to take a look!_).
# 
# To explore the feedback system, we'll start with a simple example of a coding problem.  Follow the following steps in order:
# 1. Run the code cell below without making any edits.  It will show the following output: 
# > <font color='#ccaa33'>Check:</font> When you've updated the starter code, `check()` will tell you whether your code is correct. You need to update the code that creates variable `one`
# 
#     This means you need to change the code to set the variable `one` to something other than the blank provided below (`____`).
# 
# 
# 2. Replace the underline with a `2`, so that the line of code appears as `one = 2`.  Then, run the code cell.  This should return the following output:
# > <font color='#cc3333'>Incorrect:</font> Incorrect value for `one`: `2`
# 
#     This means we still have the wrong answer to the question.
# 
# 
# 3. Now, change the `2` to `1`, so that the line of code appears as `one = 1`. Then, run the code cell. The answer should be marked as <font color='#33cc33'>Correct</font>. You have now completed this problem!


# CELL ********************

# Fill in the line below
one = ____

# Check your answer
step_1.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# In this exercise, you were responsible for filling in the line of code that sets the value of variable `one`.  **Don't edit the code that checks your answer.**  You'll need to run the lines of code like `step_1.check()` and `step_2.check()` just as they are provided.
# 
# This problem was relatively straightforward, but for more difficult problems, you may like to receive a hint or view the official solution.  Run the code cell below now to receive both for this problem.

# CELL ********************

#step_1.hint()
#step_1.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# ## Step 2: Load the data
# 
# You are ready to get started with some data visualization!  You'll begin by loading the dataset from the previous tutorial.  
# 
# The code you need is already provided in the cell below. Just run that cell. If it shows <font color='#33cc33'>Correct</font> result, you're ready to move on!

# CELL ********************

# Path of the file to read
fifa_filepath = "/lakehouse/default/Files/DMML_Aula2/fifa.csv"

# Read the file into a variable fifa_data
fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)

# Check your answer
step_2.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Next, recall the difference between comments and executable code:
# - **Comments** are preceded by a pound sign (`#`) and contain text that appear faded and italicized.  They are completely ignored by the computer when the code is run.
# - **Executable code** is code that is run by the computer.
# 
# In the code cell below, every line is a comment:
# ```python
# # Uncomment the line below to receive a hint
# #step_2.hint()
# #step_2.solution()
# ```
# 
# If you run the code cell below without making any changes, it won't return any output.  Try this now!

# CELL ********************

# Uncomment the line below to receive a hint
#step_2.hint()
# Uncomment the line below to see the solution
#step_2.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Next, remove the pound sign before `step_2.hint()` so that the code cell above appears as follows:
# ```python
# # Uncomment the line below to receive a hint
# step_2.hint()
# #step_2.solution()
# ```
# When we remove the pound sign before a line of code, we say we **uncomment** the line.  This turns the comment into a line of executable code that is run by the computer.  Run the code cell now, which should return the <font color='#3366cc'>Hint</font> as output.
# 
# Finally, uncomment the line to see the solution, so the code cell appears as follows:
# ```python
# # Uncomment the line below to receive a hint
# step_2.hint()
# step_2.solution()
# ```
# Then, run the code cell.  You should receive both a <font color='#3366cc'>Hint</font> and the <font color='#33cc99'>Solution</font>.
# 
# If at any point you're having trouble with coming up with the correct answer to a problem, you are welcome to obtain either a hint or the solution before completing the cell.  (So, you don't need to get a <font color='#33cc33'>Correct</font> result before running the code that gives you a <font color='#3366cc'>Hint</font> or the <font color='#33cc99'>Solution</font>.)
# 
# ## Step 3: Plot the data
# 
# Now that the data is loaded into the notebook, you're ready to visualize it!  
# 
# Run the next code cell without changes to make a line chart.  The code may not make sense yet - you'll learn all about it in the next tutorial!


# CELL ********************

# Set the width and height of the figure
plt.figure(figsize=(16,6))

# Line chart showing how FIFA rankings evolved over time
sns.lineplot(data=fifa_data)

# Check your answer
step_3.a.check()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Some questions won't require you to write any code.  Instead, you'll interpret visualizations.
# 
# As an example, consider the question: Considering only the years represented in the dataset, which countries spent at least 5 consecutive years in the #1 ranked spot?
# 
# To receive a <font color='#3366cc'>Hint</font>, uncomment the line below, and run the code cell.

# CELL ********************

#_COMMENT_IF(PROD)_
step_3.b.hint()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Once you have an answer, check the <font color='#33cc99'>Solution</font> to get credit for completing the problem and to ensure your interpretation is right.

# CELL ********************

# Check your answer (Run this code cell to receive credit!)
step_3.b.solution()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# MARKDOWN ********************

# Congratulations - you have completed your first coding exercise!
# 
# # Keep going
# 
# Move on to learn to create your own **[line charts]** with a new dataset.
