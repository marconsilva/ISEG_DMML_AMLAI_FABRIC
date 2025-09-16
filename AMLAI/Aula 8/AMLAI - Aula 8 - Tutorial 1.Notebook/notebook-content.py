# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "ac2a3e12-25ec-4038-a17a-4c059b361903",
# META       "default_lakehouse_name": "LakehouseGenAI",
# META       "default_lakehouse_workspace_id": "3d979b2c-af67-4dbe-963d-0c048b6b3998",
# META       "known_lakehouses": [
# META         {
# META           "id": "0ebbea0e-2e30-43d6-a275-80b6a242a5a6"
# META         },
# META         {
# META           "id": "ac2a3e12-25ec-4038-a17a-4c059b361903"
# META         },
# META         {
# META           "id": "a5591839-f387-4a67-a52e-dac9b3ea21b0"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# # Introduction
# 
# Large Language Models (LLMs) such as OpenAI's ChatGPT are powerful tools, but their effectiveness for business applications and meeting customer needs greatly improves when customized with specific data using Generative AI (GenAI) solutions. Without this customization, LLMs may not deliver optimal results tailored to the requirements and expectations of businesses and their customers.
# 
# One straightforward approach to enhance the results is to manually integrate specific information into prompts. For more advanced improvements, fine-tuning LLMs with custom data proves effective. This notebook demonstrates the Retrieval Augmented Generation (RAG) strategy, which supplements LLMs with dynamically retrieved and relevant information (e.g., business-specific data) to enrich their knowledge.
# 
# Implementing RAG involves methods such as web searching or utilizing specific APIs. An effective approach is utilizing a Vector Search Index to efficiently explore unstructured text data. The Vector Index searches through a database of text chunks and ranks them based on how closely they match the meaning of the user's question or query. Since full documents or articles are usually too large to embed directly into a vector, they are typically split into smaller chunks. These smaller chunks are then indexed in systems like Azure AI Search, making it easier to retrieve relevant information efficiently.
# 
# In this tutorial we will explore on how to run open ai inside Microsoft Fabric.

# MARKDOWN ********************

# 
# #### Use OpenAI in Microsoft Fabric


# CELL ********************

from synapse.ml.mlflow import get_mlflow_env_config

mlflow_env_configs = get_mlflow_env_config()
access_token = mlflow_env_configs.driver_aad_token

prebuilt_AI_base_url = mlflow_env_configs.workload_endpoint + "cognitive/openai/"
print("workload endpoint for OpenAI: \n" + prebuilt_AI_base_url)


deployment_name = "text-davinci-003" # deployment name could be `text-davinci-003` or `code-cushman-002`
openai_url = prebuilt_AI_base_url + f"openai/deployments/{deployment_name}/completions?api-version=2022-12-01"
print("The full uri of Completions is: ", openai_url)

post_headers = {
    "Content-Type" : "application/json",
    "Authorization" : "Bearer {}".format(access_token)
}

post_body = {
    "prompt": "empty prompt, need to fill in the content before the request",
}

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import json
import uuid
import requests
from pprint import pprint



def get_model_response_until_empty(prompt:str, openai_url:str):
    post_body["prompt"] = ""
    pr_aux = prompt

    while True:
        post_body["prompt"] = post_body["prompt"] + prompt
        response = requests.post(openai_url, headers=post_headers, json=post_body)
        if response.status_code == 200:
            prompt = response.json()["choices"][0]["text"]
            if len(prompt) == 0:
                result = post_body["prompt"]
                break
        else:
            print(response.headers)
            result = response.content
            break

    result = result[len(pr_aux):].strip()
    return result, response.status_code


def printresult(openai_url:str, response_code:int, prompt:str, result:str):
    print("==========================================================================================")
    print("| Post URI        |", openai_url)
    print("------------------------------------------------------------------------------------------")
    print("| Response Status |", response_code)
    print("------------------------------------------------------------------------------------------")
    print("| OpenAI Input    |\n", prompt)
    print("------------------------------------------------------------------------------------------")
    print("| OpenAI Output   |\n", result)
    print("==========================================================================================")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df = spark.sql("SELECT * FROM LakehouseGenAI.fine_food_reviews_1k")
display(df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

value_text_id_1000 = df.filter(df['id'] == 1000).select('text').collect()[0][0]
print(value_text_id_1000)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

prompt_agent = "Get me a summary with maximum 7 words from the settence bellow about restaurants and food review: "
prompt = prompt_agent + value_text_id_1000 
result, status = get_model_response_until_empty(prompt=prompt, openai_url=openai_url)
printresult(openai_url=openai_url, response_code=status, prompt=prompt, result=result)
summary = result

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

prompt_agent = "Get me a sentiment value from the settence bellow, it should be an int number between 1 and 5: "
prompt = prompt_agent + value_text_id_1000 
result, status = get_model_response_until_empty(prompt=prompt, openai_url=openai_url)
printresult(openai_url=openai_url, response_code=status, prompt=prompt, result=result)
sentiment = result

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import when
df = df.withColumn("score", when(df["id"] == 1000, sentiment).otherwise(df["score"]))
df = df.withColumn("summary", when(df["id"] == 1000, summary).otherwise(df["summary"]))
display(df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
