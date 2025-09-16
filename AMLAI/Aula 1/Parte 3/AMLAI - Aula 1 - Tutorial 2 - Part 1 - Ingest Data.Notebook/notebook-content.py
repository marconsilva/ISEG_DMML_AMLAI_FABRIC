# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# # INTRODUCTION
# 
# The notebooks in this tutorial are part of this Microsoft Fabric Data science Tutorial:
# 
# [Data Science in Fabric](https://learn.microsoft.com/en-us/fabric/data-science/tutorial-data-science-ingest-data)
# 
# So you should be looking to that while doing this tutorial.
# 
# 
# This set of tutorials demonstrates a sample end-to-end scenario in the Fabric data science experience. You implement each step from data ingestion, cleansing, and preparation, to training machine learning models and generating insights, and then consume those insights using visualization tools like Power BI.
# 


# MARKDOWN ********************

# # Part 1: Ingest data into a Microsoft Fabric lakehouse using Apache Spark
# 
# In this tutorial, you'll ingest data into Fabric lakehouses in delta lake format. Some important terms to understand:
# 
# * **Lakehouse** -- A lakehouse is a collection of files/folders/tables that represent a database over a data lake used by the Spark engine and SQL engine for big data processing and that includes enhanced capabilities for ACID transactions when using the open-source Delta formatted tables.
# 
# * **Delta Lake** - Delta Lake is an open-source storage layer that brings ACID transactions, scalable metadata management, and batch and streaming data processing to Apache Spark. A Delta Lake table is a data table format that extends Parquet data files with a file-based transaction log for ACID transactions and scalable metadata management.

# MARKDOWN ********************

# ## Prerequisites
# 
# - [Add a lakehouse](https://aka.ms/fabric/addlakehouse) to this notebook. You will be downloading data from a public blob, then storing the data in the lakehouse. 

# MARKDOWN ********************

# ## Bank churn data

# MARKDOWN ********************

# 
# The dataset contains churn status of 10,000 customers. It also includes attributes that could impact churn such as:
# 
# * Credit score
# * Geographical location (Germany, France, Spain)
# * Gender (male, female)
# * Age
# * Tenure (years of being bank's customer)
# * Account balance
# * Estimated salary
# * Number of products that a customer has purchased through the bank
# * Credit card status (whether a customer has a credit card or not)
# * Active member status (whether an active bank's customer or not)
# 
# The dataset also includes columns such as row number, customer ID, and customer surname that should have no impact on customer's decision to leave the bank. 
# 
# The event that defines the customer's churn is the closing of the customer's bank account. The column `exited` in the dataset refers to customer's abandonment. There isn't much context available about these attributes so you have to proceed without having background information about the dataset. The aim is to understand how these attributes contribute to the `exited` status.
# 
# Example rows from the dataset:
# 
# |"CustomerID"|"Surname"|"CreditScore"|"Geography"|"Gender"|"Age"|"Tenure"|"Balance"|"NumOfProducts"|"HasCrCard"|"IsActiveMember"|"EstimatedSalary"|"Exited"|
# |---|---|---|---|---|---|---|---|---|---|---|---|---|
# |15634602|Hargrave|619|France|Female|42|2|0.00|1|1|1|101348.88|1|
# |15647311|Hill|608|Spain|Female|41|1|83807.86|1|0|1|112542.58|0|


# MARKDOWN ********************

# ### Download dataset and upload to lakehouse
# 
# > [!TIP]
# > By defining the following parameters, you can use this notebook with different datasets easily.


# CELL ********************

IS_CUSTOM_DATA = False  # if TRUE, dataset has to be uploaded manually

DATA_ROOT = "/lakehouse/default"
DATA_FOLDER = "Files/churn"  # folder with data files
DATA_FILE = "churn.csv"  # data file name

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# This code downloads a publicly available version of the dataset and then stores it in a Fabric lakehouse.
# 
# > [!IMPORTANT]
# > **Make sure you [add a lakehouse](https://aka.ms/fabric/addlakehouse) to the notebook before running it. Failure to do so will result in an error.**

# CELL ********************

import os, requests
if not IS_CUSTOM_DATA:
# Using synapse blob, this can be done in one line

# Download demo data files into lakehouse if not exist
    remote_url = "https://synapseaisolutionsa.blob.core.windows.net/public/bankcustomerchurn"
    file_list = [DATA_FILE]
    download_path = f"{DATA_ROOT}/{DATA_FOLDER}/raw"

    if not os.path.exists("/lakehouse/default"):
        raise FileNotFoundError(
            "Default lakehouse not found, please add a lakehouse and restart the session."
        )
    os.makedirs(download_path, exist_ok=True)
    for fname in file_list:
        if not os.path.exists(f"{download_path}/{fname}"):
            r = requests.get(f"{remote_url}/{fname}", timeout=30)
            with open(f"{download_path}/{fname}", "wb") as f:
                f.write(r.content)
    print("Downloaded demo data files into lakehouse.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Next step
# 
# You'll use the data you just ingested in [Part 2: Explore and cleanse data](https://learn.microsoft.com/fabric/data-science/tutorial-data-science-explore-notebook).
