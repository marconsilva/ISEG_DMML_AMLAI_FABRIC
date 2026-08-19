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

from synapse.ml.fabric.service_discovery import get_fabric_env_config
from synapse.ml.fabric.token_utils import TokenUtils

fabric_env_config = get_fabric_env_config().fabric_env_config


deployment_name = "gpt-5.1"
openai_url = (
    f"{fabric_env_config.ml_workload_endpoint}cognitive/openai/openai/deployments/"
    f"{deployment_name}/chat/completions?api-version=2024-02-15-preview"
)
print("The full URI of Chat Completions is:", openai_url)

"""
Legacy authentication placeholder retained from the original export.
access_token = ""
post_headers = {
    "Authorization": TokenUtils().get_openai_auth_header(),
    "Content-Type" : "application/json",
    "Authorization" : "Bearer {}".format(access_token)
    , "Authorization": TokenUtils().get_openai_auth_header(),
}

post_body = {
    "prompt": "empty prompt, need to fill in the content before the request",
}
"""

post_headers = {
    "Authorization": TokenUtils().get_openai_auth_header(),
    "Content-Type": "application/json",
}

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import json
import re
import uuid
import requests
from pprint import pprint



def get_model_response_until_empty(prompt:str, openai_url:str):
    post_body = {
        "messages": [
            {"role": "system", "content": "Follow the user's instructions precisely."},
            {"role": "user", "content": prompt},
        ]
    }
    response = requests.post(
        openai_url,
        headers=post_headers,
        json=post_body,
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()["choices"][0]["message"]["content"].strip()
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


_SCORE_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}


def extract_score(content):
    """Parse an explicitly labelled 1-5 score without guessing from other digits."""
    text = str(content).strip()
    fenced_json = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    json_text = fenced_json.group(1) if fenced_json else text

    try:
        payload = json.loads(json_text)
    except (TypeError, ValueError):
        payload = None

    if isinstance(payload, dict):
        for key in ("score", "rating", "stars", "sentiment"):
            if key in payload:
                value = payload[key]
                if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 5:
                    return value
                if isinstance(value, str):
                    text = f"{key}: {value}"
                break

    value_pattern = r"([1-5]|one|two|three|four|five)"
    candidates = []
    patterns = [
        rf"^\s*{value_pattern}\s*(?:stars?|/5|out\s+of\s+5)?[.!]?\s*$",
        rf"^\s*(?:the\s+)?(?:score|rating|stars?|sentiment)\s*(?:is|:|=|-)\s*"
        rf"{value_pattern}\s*(?:stars?|/5|out\s+of\s+5)?"
        rf"(?:[.!]|\s*[-,;:]\s+.*|\s+because\s+.*)?\s*$",
    ]
    for line in text.splitlines():
        for pattern in patterns:
            match = re.fullmatch(pattern, line, re.IGNORECASE)
            if match:
                token = match.group(1).lower()
                candidates.append(int(token) if token.isdigit() else _SCORE_WORDS[token])

    sentence = re.fullmatch(
        rf"\s*I(?:\s+would|'d)?\s+(?:give(?:\s+it)?|rate(?:\s+it)?)\s+"
        rf"{value_pattern}\s*(?:stars?|/5|out\s+of\s+5)?"
        rf"(?:[.!]|\s+because\s+.*)?\s*",
        text,
        re.IGNORECASE,
    )
    if sentence:
        token = sentence.group(1).lower()
        candidates.append(int(token) if token.isdigit() else _SCORE_WORDS[token])

    unique_scores = set(candidates)
    if len(unique_scores) != 1:
        raise ValueError(f"Expected one explicit score from 1 to 5, got: {content!r}")
    return unique_scores.pop()

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

prompt_agent = (
    'Rate the sentiment of the sentence below from 1 to 5. '
    'Return only JSON in the form {"score": 3}: '
)
prompt = prompt_agent + value_text_id_1000 
result, status = get_model_response_until_empty(prompt=prompt, openai_url=openai_url)
printresult(openai_url=openai_url, response_code=status, prompt=prompt, result=result)
sentiment = extract_score(result)

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
