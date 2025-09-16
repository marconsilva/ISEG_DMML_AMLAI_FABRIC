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
# META       "default_lakehouse_workspace_id": "03f3982f-785f-4a2f-8ec0-4be54060ee7b",
# META       "known_lakehouses": [
# META         {
# META           "id": "a5591839-f387-4a67-a52e-dac9b3ea21b0"
# META         }
# META       ]
# META     },
# META     "environment": {
# META       "environmentId": "49550b50-361f-4fba-99fc-0a7290da68b6",
# META       "workspaceId": "03f3982f-785f-4a2f-8ec0-4be54060ee7b"
# META     }
# META   }
# META }

# MARKDOWN ********************

# # Building Retrieval Augmented Generation in Fabric Using Built-in OpenAI: A Step-by-Step Guide

# MARKDOWN ********************

# ## Introduction
# 
# Large Language Models (LLMs) such as OpenAI's ChatGPT are powerful tools, but their effectiveness for business applications and meeting customer needs greatly improves when customized with specific data using Generative AI (GenAI) solutions. Without this customization, LLMs may not deliver optimal results tailored to the requirements and expectations of businesses and their customers. 
# 
# One straightforward approach to enhance the results is to manually integrate specific information into prompts. For more advanced improvements, fine-tuning LLMs with custom data proves effective. This notebook demonstrates the Retrieval Augmented Generation (RAG) strategy, which supplements LLMs with dynamically retrieved and relevant information (e.g., business-specific data) to enrich their knowledge.
# 
# Implementing RAG involves methods such as web searching or utilizing specific APIs. An effective approach is utilizing a Vector Search Index to efficiently explore unstructured text data. The Vector Index searches through a database of text chunks and ranks them based on how closely they match the meaning of the user's question  or query. Since full documents or articles are usually too large to embed directly into a vector, they are typically split into smaller chunks. These smaller chunks are then indexed in systems like Azure AI Search, making it easier to retrieve relevant information efficiently.
# 
# <img src="https://appliedaipublicdata.blob.core.windows.net/cmuqa-08-09/output/fabric_guidance_genai_synapseml_openai.png" style="width:1000px;"/>
# 
# This tutorial provides a quickstart guide to use Fabric for building RAG applications. The main steps in this tutorial are as following:
# 
# 1. Set up Azure AI Search Services
# 2. Load and manipulate the data from [CMU's QA dataset](https://www.cs.cmu.edu/~ark/QA-data/) of Wikipedia articles
# 3. Chunk the data by leveraging Spark pooling for efficient processing
# 4. Create embeddings using fabric's built-in [Azure OpenAI Services through Synapse ML](https://learn.microsoft.com/en-us/fabric/data-science/ai-services/how-to-use-openai-sdk-synapse?tabs=synapseml)
# 5. Create a Vector Index using [Azure AI Search](https://aka.ms/what-is-azure-search)
# 6. Generate answers based on the retrieved context using fabrics built-in [Azure OpenAI through python SDK](https://learn.microsoft.com/en-us/fabric/data-science/ai-services/how-to-use-openai-sdk-synapse?tabs=python)
# 
# 
# ## Prerequisites
# 
# You need the following services to run this notebook.
# 
# - [Microsoft Fabric](https://aka.ms/fabric/getting-started) with F64 Capacity 
# - [Add a lakehouse](https://aka.ms/fabric/addlakehouse) to this notebook. You will download data from a public blob, then store the data in the lakehouse resource.
# - [Azure AI Search](https://aka.ms/azure-ai-search)


# MARKDOWN ********************

# ## Step 1: Overview of Azure Setup
# 
# In this tutorial, we will benefit from Fabric's built-in Azure Openai Service that requires no keys. You only need to run the cell below to enforce required synapse ml configuration. 

# CELL ********************

# MAGIC %%configure -f
# MAGIC {
# MAGIC   "name": "synapseml",
# MAGIC   "conf": {
# MAGIC       "spark.jars.packages": "com.microsoft.azure:synapseml_2.12:0.0.0-1605-2ae32b04-20240614-1828-SNAPSHOT",
# MAGIC       "spark.jars.repositories": "https://mmlspark.azureedge.net/maven",
# MAGIC       "spark.jars.excludes": "org.scala-lang:scala-reflect,org.apache.spark:spark-tags_2.12,org.scalactic:scalactic_2.12,org.scalatest:scalatest_2.12,com.fasterxml.jackson.core:jackson-databind",
# MAGIC       "spark.sql.catalog.pbi": "com.microsoft.azure.synapse.ml.powerbi.PowerBICatalog",      
# MAGIC       "spark.yarn.user.classpath.first": "true",
# MAGIC       "spark.sql.parquet.enableVectorizedReader": "false"
# MAGIC   }
# MAGIC }

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #### Set up Azure AI Search Keys
# 
# Once you have an Azure subscription, you can create an Azure AI Search Service by following the instructions [here](https://aka.ms/azure-ai-search).
# 
# You may choose a free tier for the Azure AI Search Service, which allows you to have 3 indexes and 50 MB of storage. The free tier is sufficient for this tutorial. You will need to select a subscription, set up a resource group, and name the service. Once configured, obtain the keys to specify as `aisearch_api_key`. Please complete the details for `aisearch_index_name`, etc in the following. 
# 
# <img src="https://appliedaipublicdata.blob.core.windows.net/cmuqa-08-09/output/Azure_AI_Search_Free_Tier.png" style="width:800px;"/>


# CELL ********************

# Setup key accesses to Azure AI Search
aisearch_index_name = "indexmsfb" # TODO: Create a new index name: must only contain lowercase, numbers, and dashes
aisearch_api_key = "" # TODO: Fill in your API key from Azure AI Search
aisearch_endpoint = "https://gptkb-w3rdooxxajzsm.search.windows.net" # TODO: Provide the url endpoint for your created Azure AI Search 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# After setting up your Azure AI Search Keys, you must import required libraries from [Spark](https://spark.apache.org/), [SynapseML](https://aka.ms/AboutSynapseML), [Azure Search](https://aka.ms/azure-search-libraries), and OpenAI. 
# 
# Make sure to use the `environment.yaml` from the same location as this notebook file to upload into Fabric to create, save, and publish a [Fabric environment](https://aka.ms/fabric/create-environment). Then select the newly created environment before running the cell below for imports.

# CELL ********************

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import os, requests, json, warnings

from datetime import datetime, timedelta
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from pyspark.sql import functions as F
from pyspark.sql.functions import to_timestamp, current_timestamp, concat, col, split, explode, udf, monotonically_increasing_id, when, rand, coalesce, lit, input_file_name, regexp_extract, concat_ws, length, ceil
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, ArrayType, FloatType
from pyspark.sql import Row
import pandas as pd
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import (
    VectorizedQuery,
)
from azure.search.documents.indexes.models import (  
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,   
    SemanticConfiguration,  
    SemanticPrioritizedFields,
    SemanticField,  
    SemanticSearch,
    VectorSearch, 
    HnswAlgorithmConfiguration,
    HnswParameters,  
    VectorSearchProfile,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
)

from synapse.ml.featurize.text import PageSplitter
from synapse.ml.services.openai import OpenAIEmbedding
from synapse.ml.services.openai import OpenAIChatCompletion
import ipywidgets as widgets  
from IPython.display import display as w_display
import openai

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Step 2: Load the data into the Lakehouse and Spark

# MARKDOWN ********************

# #### Dataset
# 
# The Carnegie Mellon University Question-Answer dataset version 1.2 is a corpus of Wikipedia articles, manually-generated factual questions based on the articles, and manually-generated answers. The data is hosted on an Azure blob storage under the same license [GFDL](http://www.gnu.org/licenses/fdl.html). For simplicity, the data is cleaned up and refined into a single structured table with the following fields.
# 
# - ArticleTitle: the name of the Wikipedia article from which questions and answers initially came.
# - Question: manually generated question based on article
# - Answer: manually generated answer based on question and article
# - DifficultyFromQuestioner: prescribed difficulty rating for the question as given to the question-writer
# - DiffuctlyFromAnswerer: Difficulty rating assigned by the individual who evaluated and answered the question, which may differ from the difficulty from DifficultyFromQuestioner
# - ExtractedPath: path to original article. There may be more than one Question-Answer pair per article
# - text: cleaned wikipedia artices
# 
# For more information about the license, please download a copy of the license named `LICENSE-S08,S09` from the same location.
# 
# ##### History and Citation
# 
# The dataset used for this notebook requires the following citation:
# 
#     CMU Question/Answer Dataset, Release 1.2
# 
#     8/23/2013
# 
#     Noah A. Smith, Michael Heilman, and Rebecca Hw
# 
#     Question Generation as a Competitive Undergraduate Course Project
# 
#     In Proceedings of the NSF Workshop on the Question Generation Shared Task and Evaluation Challenge, Arlington, VA, September 2008. 
#     Available at: http://www.cs.cmu.edu/~nasmith/papers/smith+heilman+hwa.nsf08.pdf
# 
#     Original dataset acknowledgements:
#     This research project was supported by NSF IIS-0713265 (to Smith), an NSF Graduate Research Fellowship (to Heilman), NSF IIS-0712810 and IIS-0745914 (to Hwa), and Institute of Education Sciences, U.S. Department of Education R305B040063 (to Carnegie Mellon).
# 
#     cmu-qa-08-09 (modified verison)
# 
#     6/12/2024
# 
#     Amir Jafari, Alexandra Savelieva, Brice Chung, Hossein Khadivi Heris, Journey McDowell
# 
#     Released under same license GFDL (http://www.gnu.org/licenses/fdl.html)
#     All the GNU license applies to the dataset in all copies.
# 
#     

# CELL ********************

# Publicly hosted refined dataset inspired by https://www.cs.cmu.edu/~ark/QA-data/
storage_account_name = "appliedaipublicdata"
container_name = "cmuqa-08-09"

wasbs_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/output/part-00000-c258b030-b04e-4f9d-b887-c85225af4332-c000.snappy.parquet"  

# Save as a delta lake, parquet table to Tables section of the default lakehouse
spark_df = spark.read.parquet(wasbs_path)

display(spark_df)

spark_df.write.mode("overwrite").format("delta").saveAsTable("dbo.cmu_qa_08_09")


# Read parquet table from default lakehouse into spark dataframe
#df_dataset = spark.sql("SELECT * FROM cmu_qa_08_09")
#display(df_dataset)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# The original dataset is divided into student Semesters S08, S09, and S10. Each semester contains multiple sets, and each set comprises approximately 10 Wikipedia articles. As illustrated earlier, due to varying licenses, the entire datasets are consolidated into a single table encompassing S08 and S09, omitting S10. For sake of simplicity in demonstration, this tutorial will specifically highlight sets 1 and 2 within S08. The primary focus areas will be `wildlife` and `countries`.

# CELL ********************

# Filter the DataFrame to include only the specified paths
df_selected = df_dataset.filter((col("ExtractedPath").like("S08/data/set1/%")) | (col("ExtractedPath").like("S08/data/set2/%")))

# Select only the required columns
filtered_df = df_selected.select('ExtractedPath', 'ArticleTitle', 'text')

# Drop duplicate rows based on ExtractedPath, ArticleTitle, and text
df_wiki = filtered_df.dropDuplicates(['ExtractedPath', 'ArticleTitle', 'text'])

# Show the result
display(df_wiki)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Step 3: Chunk the Text 

# MARKDOWN ********************

# When large documents are inputted into the LLMs, it needs to extract the most important information to answer user queries. Chunking involves breaking down large text into smaller segments or chunks. In the RAG context, embedding smaller chunks rather than entire documents for the knowledge base means retrieving only the most relevant chunks in response to a user's query. This approach reduces input tokens and provides more focused context for the LLM to process.
# 
# To perform chunking, you should use the `PageSplitter` implementation from the `SynapseML` library for distributed processing. Adjusting the page length parameters (in characters) is crucial for optimizing performance based on the text size supported by the language model and the number of chunks selected as context for the conversational bot. For demonstration, a page length of 4000 characters is recommended.

# CELL ********************

ps = (
    PageSplitter()
    .setInputCol("text")
    .setMaximumPageLength(4000)
    .setMinimumPageLength(3000)
    .setOutputCol("chunks")
)

df_splitted = ps.transform(df_wiki) 
display(df_splitted.limit(10)) 

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Note that each row can contain multiple chunks from the same document represented as a vector. The function `explode` distributes and duplicates the vector's content across several rows.

# CELL ********************

df_chunks = df_splitted.select('ExtractedPath', 'ArticleTitle', 'text', explode(col("chunks")).alias("chunk"))
display(df_chunks)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Now, you will add a unique id for each row. 

# CELL ********************

df_chunks_id = df_chunks.withColumn("Id", monotonically_increasing_id())
display(df_chunks_id)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# #

# MARKDOWN ********************

# ## Step 4: Create Embeddings

# MARKDOWN ********************

# In RAG, embedding refers to incorporating relevant chunks of information from documents into the model's knowledge base. These chunks are chosen based on their relevance to potential user queries, allowing the model to retrieve specific and targeted information rather than entire documents. Embedding helps optimize the retrieval process by providing concise and pertinent context for generating accurate responses to user inputs. In this section, we will use SynapseML Library to obtain embeddings for each chunk of text.


# CELL ********************

Embd = (
    OpenAIEmbedding()
    .setDeploymentName('text-embedding-ada-002') # set deployment_name as text-embedding-ada-002
    .setTextCol("chunk")
    .setErrorCol("error")    
    .setOutputCol("Embedding")
)
df_embeddings = Embd.transform(df_chunks_id)
display(df_embeddings)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Step 5: Create Vector Index with Azure AI Search 

# MARKDOWN ********************

# In RAG, creating a vector index helps quickly retrieve the most relevant information for user queries. By organizing document chunks into a vector space, RAG can match and generate responses based on similar content rather than just keywords. This makes the responses more accurate and meaningful, improving how well the system understands and responds to user inputs.
# 
# In the next steps, you will set up a search index in Azure AI Search that integrates both semantic and vector search capabilities. You can begin by initializing the `SearchIndexClient` with the required endpoint and API key. Then, define the data structure using a list of fields, specifying their types and attributes. The `Chunk` field will hold the text to be retrieved, while the `Embedding` field will facilitate vector-based searches. Additional fields like `ArticleTitle` and `ExtractedPath` can be included for filtering purposes. For custom datasets, you can adjust the fields as necessary.
# 
# For vector search, configure the Hierarchical Navigable Small Worlds (HNSW) algorithm by specifying its parameters and creating a usage profile. You can also set up semantic search by defining a configuration that emphasizes specific fields for improved relevance. Finally, create the search index with these configurations and utilize the client to create or update the index, ensuring it supports advanced search operations. 
# 
# Note that while this tutorial focuses on vector search, Azure Search offers text search, filtering, and semantic ranking capabilities that are beneficial for various applications.

# MARKDOWN ********************

# > [!TIP]
# > You can skip the following details. You will use the Python SDK for Azure AI Search to create a new Vector Index. This index will include fields for `Chunk`, which holds the text to be retrieved, and `Embedding`, generated from the OpenAI embedding model. Additional searchable fields like `ArticleTitle` and `ExtractedPath` are useful in this dataset, but you can customize your own dataset by adding or removing fields as needed.

# CELL ********************

index_client = SearchIndexClient(
    endpoint=aisearch_endpoint,
    credential=AzureKeyCredential(aisearch_api_key),
)
fields = [
    SimpleField(name="Id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
    SearchableField(name="ArticleTitle", type=SearchFieldDataType.String, filterable=True),
    SearchableField(name="ExtractedPath", type=SearchFieldDataType.String, filterable=True),
    SearchableField(name="Chunk", type=SearchFieldDataType.String, searchable=True),
    SearchField(name="Embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="my-vector-config"
    ),
]

vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="myHnsw",
            kind=VectorSearchAlgorithmKind.HNSW,
            parameters=HnswParameters(
                m=4,
                ef_construction=400,
                ef_search=500,
                metric=VectorSearchAlgorithmMetric.COSINE
            )
        )
    ],
    profiles=[
        VectorSearchProfile(
            name="my-vector-config",
            algorithm_configuration_name="myHnsw",
        ),
    ]
)

# Note: Useful for reranking 
semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        title_field=SemanticField(field_name="ArticleTitle"),
        prioritized_content_fields=[SemanticField(field_name="Chunk")]
    )
)

# Create the semantic settings with the configuration
semantic_search = SemanticSearch(configurations=[semantic_config])

# Create the search index with the semantic settings
index = SearchIndex(
    name=aisearch_index_name,
    fields=fields,
    vector_search=vector_search,
    semantic_search=semantic_search
)
result = index_client.create_or_update_index(index)
print(f' {result.name} created')


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# The following code defines a User Defined Function (UDF) named `insertToAISearch` that inserts data into an Azure AI Search index. The integration between Azure and Spark offers a significant advantage for handling large datasets efficiently. Although the current dataset is not particularly large, employing Spark User-Defined Functions (UDFs) ensures readiness for future scalability. UDFs enable the creation of custom functions that can process Spark DataFrames, enhancing Spark's capabilities. This UDF, annotated with `@udf(returnType=StringType())`, specifies the return type as a string. The function takes five parameters: `Id`, `ArticleTitle`, `ExtractedPath`, `Chunk`, and `Embedding`. It constructs a URL for the Azure AI Search API, incorporating the search service name and index name. The function then creates a payload in JSON format, including the document fields and specifying the search action as `upload`. The headers are set to include the content type and the API key for authentication. A POST request is sent to the constructed URL with the headers and payload, and the response from the server is printed. This function facilitates the uploading of documents to the Azure AI Search index. Please make sure to include the fields specified in the previous section for your own dataset.

# CELL ********************

@udf(returnType=StringType())
def insertToAISearch(Id, ArticleTitle, ExtractedPath, Chunk, Embedding):
    url = f"{aisearch_endpoint}/indexes/{aisearch_index_name}/docs/index?api-version=2023-11-01"

    payload = json.dumps(
        {
            "value": [
                {
                    "Id": str(Id),
                    "ArticleTitle": ArticleTitle,
                    "ExtractedPath": ExtractedPath,
                    "Chunk": Chunk, 
                    "Embedding": Embedding.tolist(),
                    "@search.action": "upload",
                },
            ]
        }
    )

    headers = {
        "Content-Type": "application/json",
        "api-key": aisearch_api_key,
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)

    if response.status_code == 200 or response.status_code == 201:
        return "Success"
    else:
        return response.text

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# In the following, you will be using the previously defined UDF `insertToAISearch` to upload data from a DataFrame to the Azure AI Search index. The DataFrame `df_embeddings` contains fields such as `Id`, `ArticleTitle`, `ExtractedPath`, `Chunk`, and `Embedding`.
# 
# You apply the `insertToAISearch` function to each row to add a new column named `errorAISearch` to `df_embeddings`. This column captures responses from the Azure AI Search API, allowing you to check for any upload errors. This error checking ensures that each document is successfully uploaded to the search index.
# 
# Finally, you use the `display` function to examine the modified DataFrame `df_embeddings_ingested` visually and verify the processing accuracy.

# CELL ********************

df_embeddings_ingested = df_embeddings.withColumn(
    "errorAISearch",
    insertToAISearch(
        df_embeddings["Id"],
        df_embeddings["ArticleTitle"],
        df_embeddings["ExtractedPath"],
        df_embeddings["Chunk"],
        df_embeddings["Embedding"]
    ),
)

display(df_embeddings_ingested)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# You can now proceed to perform sanity checks to ensure the data has been correctly uploaded to the Azure AI Search index. First, count the number of successful uploads by filtering the DataFrame for rows where `errorAISearch` is "Success" and using the count method to determine the total. Next, identify unsuccessful uploads by filtering for rows containing errors in `errorAISearch` and count these occurrences. Print the counts of successful and unsuccessful uploads to summarize the results. If there are any unsuccessful uploads, use the show method to display details of those rows. This allows you to inspect and address any issues, ensuring the upload process is validated and any necessary corrective actions are taken.

# CELL ********************

# Count the number of successful uploads
successful_uploads = df_embeddings_ingested.filter(col("errorAISearch") == "Success").count()

# Identify and display unsuccessful uploads
unsuccessful_uploads = df_embeddings_ingested.filter(col("errorAISearch") != "Success")
unsuccessful_uploads_count = unsuccessful_uploads.count()

# Display the results
print(f"Number of successful uploads: {successful_uploads}")
print(f"Number of unsuccessful uploads: {unsuccessful_uploads_count}")

# Show details of unsuccessful uploads if any
if unsuccessful_uploads_count > 0:
    unsuccessful_uploads.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Step 6: Demonstrate Retrieval Augmented Generation

# MARKDOWN ********************

# Once you've chunked, embedded, and created a vector index, the final step is to use this indexed data to find and retrieve the most relevant information based on user queries. This allows the system to generate accurate responses or recommendations by leveraging the indexed data's organization and similarity scores from the embeddings. 
# 
# In the following, you create a function for retrieving chunks of relevant Wikipedia articles from the vector index named Azure AI Search. Whenever a new question gets asked, you procedurally
# 
# - embed the question into a vector
# - retrieve the top N chunks from Azure AI Search using the vector
# - concatenate the results to get a single string

# CELL ********************

def get_context_source(question, topN=3):
    """
    Retrieves contextual information and sources related to a given question using embeddings and a vector search.  
    Parameters:  
    question (str): The question for which the context and sources are to be retrieved.  
    topN (int, optional): The number of top results to retrieve. Default is 3.  
      
    Returns:  
    List: A list containing two elements:  
        1. A string with the concatenated retrieved context.  
        2. A list of retrieved source paths.  
    """

    deployment_id = "text-embedding-ada-002"

    query_embedding = openai.Embedding.create(deployment_id=deployment_id,
                                     input=question).data[0].embedding

    vector_query = VectorizedQuery(vector=query_embedding, k_nearest_neighbors=topN, fields="Embedding")

    search_client = SearchClient(
        aisearch_endpoint,
        aisearch_index_name,
        credential=AzureKeyCredential(aisearch_api_key)
    )

    results = search_client.search(   
        vector_queries=[vector_query],
        top=topN,
    )

    retrieved_context = ""
    retrieved_sources = []
    for result in results:
        retrieved_context += result['ExtractedPath'] + "\n" + result['Chunk'] + "\n\n"
        retrieved_sources.append(result['ExtractedPath'])

    return [retrieved_context, retrieved_sources]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# You need another function to get the response from the OpenAI Chat model. This function combines the user question with the context retrieved from Azure AI Search. This example is basic and doesn't include chat history or memory. First, you initialize the chat client with the chosen model and then perform a chat completion to obtain the response. The messages have a "system" content that can be adjusted to enhance the response's tone, conciseness, and other aspects.

# CELL ********************

def get_answer(question, context):
    """  
    Generates a response to a given question using provided context and an Azure OpenAI model.  
    
    Parameters:  
        question (str): The question that needs to be answered.  
        context (str): The contextual information related to the question that will help generate a relevant response.  
    
    Returns:  
        str: The response generated by the Azure OpenAI model based on the provided question and context.  
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful chat assistant who will be provided text information for you to refer to in response."
        }
    ]

    messages.append(
        {
            "role": "user", 
            "content": question + "\n" + context,
        },
    )
    response = openai.ChatCompletion.create(
        deployment_id='gpt-35-turbo-0125', # see the note in the cell below for an alternative deployment_id.
        messages= messages,
        temperature=0,
    )

    return response.choices[0].message.content


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Note: please consult the [documentation for python SDK](https://learn.microsoft.com/en-us/fabric/data-science/ai-services/how-to-use-openai-sdk-synapse?tabs=python) for the other available deployment_ids. 

# CELL ********************

question = "How do elephants communicate over long distances?"
retrieved_context, retrieved_sources = get_context_source(question)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

answer = get_answer(question, retrieved_context)
print(answer)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# You have successfully learned how to use the tools mentioned above to embed and chunk the CMU QA dataset for your RAG application. Now that you've seen the retrieval and answering functions in action, you can create a basic ipywidget to serve as a chatbot interface.  After running the cell below, enter your question and press `Enter` to get a response from the RAG solution. Modify the text to ask a new question and press `Enter` again.
# 
# > [!Tip]
# > This RAG solution can make mistakes. Feel free to change the OpenAI model to gpt-4 or modify the system content prompt 

# CELL ********************

# Create a text box for input  
text = widgets.Text(  
    value='',  
    placeholder='Type something',  
    description='Question:',  
    disabled=False,  
    continuous_update=False,  
    layout=widgets.Layout(width='800px')  # Adjust the width as needed  
)  
  
# Create an HTML widget to display the answer  
label = widgets.HTML(  
    value='',  
    layout=widgets.Layout(width='800px')  # Adjust the width as needed  
)  
  
# Define what happens when the text box value changes  
def on_text_change(change):  
    if change['type'] == 'change' and change['name'] == 'value':  
        retrieved_context, retrieved_sources = get_context_source(change['new'])  
        label.value = f"<div style='word-wrap: break-word; line-height: 1;'>{get_answer(change['new'], retrieved_context)}</div>"  
  
text.observe(on_text_change)  
  
# Display the text box and label  
w_display(text, label)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# This concludes the Quickstart tutorial on creating a RAG application in Fabric using Fabric's built-in OpenAI endpoint. 
# 
# Fabric is a platform for unifying your company's data, empowering you to leverage knowledge for your GenAI applications effectively.
