import os

from dotenv import load_dotenv

import pandas as pd

import numpy as np

import matplotlib.colors as mcolors

import streamlit as st

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from langchain.schema import SystemMessage

from pydantic import BaseModel, Field

from typing import Type

from bs4 import BeautifulSoup

import requests

import json

from fastapi import FastAPI

from hash import *

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

# 1. Tool for search


def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text

# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("THIS CONTENT:", text)

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")


# 3. Create langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events and data. You should ask targeted questions."
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""You are a top world class researcher, who can conduct detailed research on any topic and produce facts based on results; 
            you do not hallucinate, you will try as hard as possible to gather facts & data to back up and prove the research.
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective.
            2/ If there are urls of relevant links & articles, you will scrape it to gather more information.
            3/ After scraping & search, you shall think "Are there any new information I should search & scraping based on the data I collected to increase the research quality?" If answer is yes, continue; But do not do this more than 5 iterations.
            4/ You shall not hallucinate, you should only write facts & data that you have gathered.
            5/ In the final output, You shall include all reference data & links to back up and prove your research; You should include all reference data & links to back up and prove your research.
            6/ In the final output, You shall include all reference data & links to back up and prove your research; You should include all reference data & links to back up and prove your research."""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)


#4. Use streamlit to create a web app
def main():
    global FileName
    FileName = "download.txt"
    st.set_page_config(page_title="AI Research Agent", page_icon=":robot_face:")

    st.header(":robot_face: AI Research Agent")
    st.write("As an AI Research Agent I will try my very best to answer to all of your queries!")
    query = st.text_area("Research Goal:")

    if query:
        st.write("Doing research for: ", query)

        result = agent({"input": query})
        
        st.info(result['output'])

        st.download_button('Download as TXT', str(result['output']))  
    
    st.divider()
    st.subheader("Users currently using our Research Agents:")
    df1 = pd.DataFrame({
    "col1": np.random.randn(500) / 50 + 22.36,
    "col2": np.random.randn(500) / 50 + 114.2,
    "col3": 1,
    "col4": np.random.rand(500, 4).tolist(),
    })

    print(df1)
    st.map(df1, latitude = "col1", longitude = "col2", size = "col3", color = "col4")
    st.divider()
    st.caption("Made with :white_heart: by Caleb Kan")
    st.caption("Contact: calebkan1106@gmail.com")
    

if __name__ == '__main__':
    main()


# 5. Set this as an API endpoint via FastAPI
app = FastAPI()


class Query(BaseModel):
    query: str


@app.post("/")
def researchAgent(query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content['output']
    return actual_content