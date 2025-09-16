# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# # Part 4: Score the trained model


# MARKDOWN ********************

# 
# Microsoft Fabric allows you to operationalize machine learning models with a scalable function called PREDICT, which supports batch scoring in any compute engine. You can generate batch predictions directly from a Microsoft Fabric notebook or from a given model's item page. Learn about [PREDICT](https://aka.ms/fabric-predict).  
# 
# To generate batch predictions on our test dataset, you'll use version 1 of the trained churn model. You'll load the test dataset into a spark DataFrame and create an MLFlowTransformer object to generate batch predictions. You can then invoke the PREDICT function using one of following three ways: 
# 
# - Using the Transformer API from SynapseML
# - Using the Spark SQL API
# - Using PySpark user-defined function (UDF)
# 
# ## Prerequisites
# 
# - Complete [Part 3: Train and register machine learning models](https://learn.microsoft.com/fabric/data-science/tutorial-data-science-train-models).
# - Attach the same lakehouse you used in Part 3 to this notebook.

# MARKDOWN ********************

# ## Load the test data
# 
# Load the test data that you saved in Part 3.

# CELL ********************

df_test = spark.read.format("delta").load("Tables/df_test")
display(df_test)

# MARKDOWN ********************

# ### PREDICT with the Transformer API
# 
# To use the Transformer API from SynapseML, you'll need to first create an MLFlowTransformer object.
# 
# ### Instantiate MLFlowTransformer object
# 
# The MLFlowTransformer object is a wrapper around the MLFlow model that you registered in Part 3. It allows you to generate batch predictions on a given DataFrame. To instantiate the MLFlowTransformer object, you'll need to provide the following parameters:
# 
# - The columns from the test DataFrame that you need as input to the model (in this case, you would need all of them).
# - A name for the new output column (in this case, predictions).
# - The correct model name and model version to generate the predictions (in this case, `lgbm_sm` and version 1).

# CELL ********************

from synapse.ml.predict import MLFlowTransformer

model = MLFlowTransformer(
    inputCols=list(df_test.columns),
    outputCol='predictions',
    modelName='lgbm_sm',
    modelVersion=1
)

# MARKDOWN ********************

# Now that you have the MLFlowTransformer object, you can use it to generate batch predictions.

# CELL ********************

import pandas

predictions = model.transform(df_test)
display(predictions)

# MARKDOWN ********************

# ### PREDICT with the Spark SQL API

# CELL ********************

from pyspark.ml.feature import SQLTransformer 

# Substitute "model_name", "model_version", and "features" below with values for your own model name, model version, and feature columns
model_name = 'lgbm_sm'
model_version = 1
features = df_test.columns

sqlt = SQLTransformer().setStatement( 
    f"SELECT PREDICT('{model_name}/{model_version}', {','.join(features)}) as predictions FROM __THIS__")

# Substitute "X_test" below with your own test dataset
display(sqlt.transform(df_test))

# MARKDOWN ********************

# ### PREDICT with a user-defined function (UDF)

# CELL ********************

from pyspark.sql.functions import col, pandas_udf, udf, lit

# Substitute "model" and "features" below with values for your own model name and feature columns
my_udf = model.to_udf()
features = df_test.columns

display(df_test.withColumn("predictions", my_udf(*[col(f) for f in features])))

# MARKDOWN ********************

# ## Write model prediction results to the lakehouse
# 
# Once you have generated batch predictions, write the model prediction results back to the lakehouse.  

# CELL ********************

# Save predictions to lakehouse to be used for generating a Power BI report
table_name = "customer_churn_test_predictions"
predictions.write.format('delta').mode("overwrite").save(f"Tables/{table_name}")
print(f"Spark DataFrame saved to delta table: {table_name}")


# MARKDOWN ********************

# ## Next step
# 
# Use these predictions you just saved to [create a report in Power BI](https://learn.microsoft.com/fabric/data-science/tutorial-data-science-create-report).
