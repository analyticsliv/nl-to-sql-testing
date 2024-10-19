import getpass
import os
from langchain_community.utilities import SQLDatabase
from langchain_google_vertexai import VertexAI
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate, MessagesPlaceholder
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from google.cloud import bigquery
from google.cloud import storage
import json
import sys
import google.auth


def text_to_analytics (question, table, google_llm, db, bq_client):

    prefix = '''You are a Bigquery SQL expert. Given an input question, first create a syntactically correct Bigquery SQL query to run, then look at the results of the query and return the answer to the input question.
You must query only the columns that are needed to answer the question. Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Along with the SQL query, return the names of the columns used in the query. Remember to return the aliases of the column names, if used. 

DO NOT use backslash (\) to escape characters. 

Below are a few examples questions along with their corresponding SQL queries and column names: '''

    suffix = ''' Only use the following tables:
{table_info}

Question: {input} 
SQL Query:
Column Names: '''


    few_shot_examples = [

    {
        "input" : "Total daily users in September 2024",
        "query" : "SELECT date, active1DayUsers AS Total_Users FROM `wex-ga4-bigquery.wex_nl_to_sql.Active_Users` WHERE date BETWEEN '20240901' AND '20240930'",
        "Column Names" : "date, Total_Users"
    },
    {
        "input" : "Total daily viewers coming from newsbreak.com in July 2024",
        "query" : "SELECT date, SUM(totalUsers) AS Total_Users FROM `wex-ga4-bigquery.wex_nl_to_sql.Session_Source` WHERE date BETWEEN '20240901' AND '20240930' AND sessionSource = 'newsbreak.com' GROUP BY date",
        "Column Names" : "date, Total_Users"
    },
    {
        "input" : "Total daily users who viewed opinion in August 2024",
        "query" : "SELECT date, SUM(totalUsers) AS Total_Users FROM `wex-ga4-bigquery.wex_nl_to_sql.Section` WHERE date BETWEEN '20240801' AND '20240831' AND SPLIT(LTRIM(unifiedPagePathScreen, '/'),'/')[0] = 'opinion' GROUP BY date",
        "Column Names" : "date, Total_Users"
    },
    {
        "input" : "Total daily users who viewed sections opinion, policy, and news in September 2024",
        "query" : "SELECT date, SUM(Opinion) AS Opinion, SUM(Policy) AS Policy, SUM(News) AS News FROM(SELECT date, CASE WHEN SPLIT(LTRIM(unifiedPagePathScreen, '/'),'/')[0] = 'opinion' THEN totalUsers ELSE 0 END AS Opinion, CASE WHEN SPLIT(LTRIM(unifiedPagePathScreen, '/'),'/')[0] = 'policy' THEN totalUsers ELSE 0 END AS Policy, CASE WHEN SPLIT(LTRIM(unifiedPagePathScreen, '/'),'/')[0] = 'news' THEN totalUsers ELSE 0 END AS News FROM `wex-ga4-bigquery.wex_nl_to_sql.Section` WHERE date BETWEEN '20240901' AND '20240930') GROUP BY date",
        "Column Names" : "date, Opinion, Policy, News"
    }
]

    example_prompt = PromptTemplate.from_template("Question: {input}\nSQL query: {query}\nColumn Names: {Column Names}")

    prompt = FewShotPromptTemplate(
    examples = few_shot_examples,
    example_prompt = example_prompt,
    prefix = prefix,
    suffix = suffix,
    input_variables = ["input", "table_info"])

    _prompt_ = PromptTemplate.from_template(prompt.format(input = question, table_info = db.table_info))

    sql_chain = _prompt_ | google_llm


    description_prompt = '''

You are expert at annotating datasets. Given a string representation of a dataset, you must do the following:

1) Generate a name for the dataset
2) Generate a description of the dataset
3) Generate a description for the fields in the dataset
4) Give the data types of the fields in the dataset

You must return an updated dictionary without any preamble or explanation.

DO NOT USE CURLY BRACES ANYWHERE IN THE OUTPUT. USE SQUARE BRACES INSTEAD.

Dataset : {input}
Description : '''

    description_prompt = PromptTemplate.from_template(description_prompt)
    describer = description_prompt | google_llm

    viz_prompt = '''

You are an expert data analyst in Python and Matplotlib. Given a description of a dataset, your job is to list down the most suitable data visualizations.
You must return the name of the visualizations without any extra information or explanation. ALWAYS return the output as a list.

DO NOT USE CURLY BRACES IN THE OUTPUT. USE SQUARE BRACES INSTEAD.

Description : {input}
Visualizations : '''

    viz_prompt = PromptTemplate.from_template(viz_prompt)
    visualizer = viz_prompt | google_llm

    function_calling_prefix = '''

You are a Python data visualization and Streamlit expert who is proficient in Matplotlib. You will be given
a dataset description and a list of visualizations. Your job is to generate appropriate Matplotlib
API calls along with the required arguments. Write the API calls such that they can be displayed on a streamlit UI. Pay attention to the fields information in the dataset descriptiion 
for arguments. ALWAYS pass data as the value of data argument.

If the Visualizations List contains more than one visualization then generate API calls such that all the listed visualizations
should be displayed at once.

DO NOT include any import statements. DO NOT include any comments.

YOU MUST NOT INCLUDE ``` AND python in the output.

YOU MUST SEPARATE EACH API CALL WITH ----.

YOU MUST STRICTLY ONLY RETURN THE API CALLS.

Below are some Dataset Descriptions and Visualization Lists along with their corresponding API calls: '''


    function_calling_suffix = ''' 

Dataset Description : {input1}
Visualizations List : {input2}
API Calls: '''

    few_shot_function_calling = [

        {"Dataset Description" : '''"name": "Daily News and Policy Engagement Data",\n"description": "This dataset tracks daily engagement metrics for news and policy content.",\n"fields": [\n"date": "Date in YYYYMMDD format",\n"News": "Number of engagements with news content",\n"Policy": "Number of engagements with policy content"\n],\n"data_types": [\n"date": "int",\n"News": "float",\n"Policy": "float"\n] \n''',
        "Visualizations List" : '["Line chart", "Stacked area chart"] \n',
        "API Calls" : '''a = plt.figure(1)
plt.plot(data['date'], data['News'], label='News')
plt.plot(data['date'], data['Policy'], label='Policy')
plt.xlabel('Date')
plt.ylabel('Engagements')
plt.legend()
st.pyplot(a)

----

b = plt.figure(2)
plt.stackplot(data['date'], data['News'], data['Policy'], labels=['News', 'Policy'])
plt.xlabel('Date')
plt.ylabel('Engagements')
plt.legend()
st.pyplot(b)'''}
]

    function_calling_example_prompt = PromptTemplate.from_template("Dataset Description: {Dataset Description}\nVisualizations List: {Visualizations List}\nAPI Calls: {API Calls}")

    function_caller_prompt = FewShotPromptTemplate(
        examples = few_shot_function_calling,
        example_prompt = function_calling_example_prompt,
        prefix = function_calling_prefix,
        suffix = function_calling_suffix,
        input_variables = ["input1", "input2"])

    try:
        
        response = sql_chain.invoke({'input' : question}).replace('```sql', '').replace('```', '')

    except:

        st.write('SQL Generation Error.')

        rows = [{'Input' : f'{question}', 'SQL' : 'SQL Generation Error.', 'Data_Description' : '', 'Visualizations' : '', 'API_Calls' : json.dumps({'0' : ''}), 'Remark' : ''}]
        bq_client.insert_rows_json(table, rows)

        st.stop()

    else:
        
        response = response.split()
        response = ' '.join(response)
        sql, column_names = response.split('Column Names')
        sql = sql.replace('SQL Query:', '').replace('SQL query:', '')
        column_names = column_names.replace(': ', '').replace(' ', '').split(',')

    # try:
    st.write(sql)
    st.write(column_names)
    data = pd.DataFrame(eval(db.run(sql)), columns = column_names)

    # except:

    #     st.write('SQL Execution Error')

    #     rows = [{'Input' : f'{question}', 'SQL' : f'{sql}', 'Data_Description' : 'SQL Execution Error.', 'Visualizations' : '', 'API_Calls' : json.dumps({'0' : ''}), 'Remark' : ''}]
    #     bq_client.insert_rows_json(table, rows)

    #     st.stop()

    # else:
        
    #     dataframe_string = pd.DataFrame(eval(db.run(sql)), columns = column_names).to_string()
        
    description = describer.invoke({'input' : dataframe_string})
    visualizations = visualizer.invoke({'input' : description})
    function_caller_prompt = PromptTemplate.from_template(function_caller_prompt.format(input1 = description, input2 = visualizations))
    function_caller = function_caller_prompt | google_llm
    calls = function_caller.invoke({'input' : 'input'})
    calls = calls.split('----')

    calls_json = {}

    for i in range(len(calls)):

        calls_json[str(i)] = calls[i]

    calls_json = json.dumps(calls_json)

    st.dataframe(data)

    try:
    
        for call in calls:

            exec(call)

    except:

        st.write('API Call Execution Error')

        rows = [{'Input' : f'{question}', 'SQL' : f'{sql}', 'Data_Description' : f'{description}', 'Visualizations' : f'{visualizations}', 'API_Calls' : f'{calls_json}', 'Remark' : ''}]
        bq_client.insert_rows_json(table, rows)

        st.stop()

    return sql, description, visualizations, calls_json

def click_button():
    
    st.session_state.clicked = True

st.set_page_config(page_title = 'AI Dashboard - Washington Examiner. (Testing Interface)')
st.header('AI Dashboard - Washington Examiner. (Testing Interface)\n\nThis application is powered by Gemini Pro')

credentials_path = 'gs://nl_to_sql_credentials/text_to_analytics.json'

storage_client = storage.Client()
bucket = storage_client.bucket('nl_to_sql_credentials')
credentials = bucket.blob('text_to_analytics.json')

project = 'wex-ga4-bigquery'
dataset = 'wex_nl_to_sql'

url = f'bigquery://{project}/{dataset}'#credentials_path={credentials_path}'
db = SQLDatabase.from_uri(url)

# credentials_, project = google.auth.default()
# st.write(credentials_)
bq_client = bigquery.Client()
table = bq_client.get_table("wex-ga4-bigquery.wex_nl_to_sql.llm_testing")

google_llm = VertexAI(model = "gemini-1.5-pro", temperature = 0.1)

# st.set_page_config(page_title = 'AI Dashboard - Washington Examiner. (Testing Interface)')
# st.header('AI Dashboard - Washington Examiner. (Testing Interface)\n\nThis application is powered by Gemini Pro')

question = st.text_input("Input: ", key = 'input')

if 'clicked' not in st.session_state:
    
    st.session_state.clicked = False

submit = st.button("Ask a question", on_click = click_button)

if st.session_state.clicked:

    sql, description, visualizations, calls_json = text_to_analytics (question, table, google_llm, db, bq_client)

    remark = st.text_input("Remark: ", key = 'remark')
    submit_remark = st.button("Please submit a remark")

    if submit_remark:

        rows = [{'Input' : f'{question}', 'SQL' : f'{sql}', 'Data_Description' : f'{description}', 'Visualizations' : f'{visualizations}', 'API_Calls' : f'{calls_json}', 'Remark' : remark}]
        bq_client.insert_rows_json(table, rows)        
