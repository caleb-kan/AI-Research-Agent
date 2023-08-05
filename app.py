import os

from dotenv import load_dotenv

from streamlit_js_eval import get_geolocation

from pymongo.mongo_client import MongoClient

import pandas as pd

import numpy as np

import matplotlib.colors as mcolors

import streamlit as st

import openai

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
from langchain.callbacks import get_openai_callback

from pydantic import BaseModel, Field

from typing import Type

from bs4 import BeautifulSoup

import requests

import json

from fastapi import FastAPI

import tweepy

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

uri = os.getenv("MONGODB_ACCESS")
cluster = MongoClient(uri)

#twitter api set up
#client = tweepy.Client(str(os.getenv("BEAR_TOKEN")), wait_on_rate_limit=True)
client = tweepy.Client(bearer_token = str(os.getenv("BEARER_TOKEN")),
                       consumer_key = str(os.getenv('TWITTER_API_KEY')),
                       consumer_secret = str(os.getenv('TWITTER_API_KEY_SECRET')),
                       access_token = str(os.getenv('ACCESS_TOKEN')),
                       access_token_secret = str(os.getenv('ACCESS_TOKEN_SECRET')),
                       wait_on_rate_limit = False)


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

# 2. Tool for search

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

# 3. Tool for scraping
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
    description = "useful when you need to get data from a website url, pass both the objective and url to the function; DO NOT make up any url, the url shall only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")

def location():
    loc = get_geolocation()
    return loc

def split_string(s, limit):
    words = s.split()
    strings = []
    current_string = words[0]
    for word in words[1:]:
        if len(current_string) + len(word) + 1 > limit:  # +1 for the space
            strings.append(current_string)
            current_string = word
        else:
            current_string += ' ' + word
    strings.append(current_string)  # append the last string
    return strings

tools = [
        Tool(
            name="Search",
            func=search,
            description="Useful for when you need to answer questions regarding all topics. You shall ask targeted questions."
        ),
        ScrapeWebsiteTool(),
    ]

memory = ConversationSummaryBufferMemory(
            memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

system_message = SystemMessage(
        content=f"""You are a top world class researcher who works for OpenAI and is cleared to receive any internal information about OpenAI, ChatGPT, GPT-3.5, and GPT-4. You will conduct extremely detailed research on any topic and produce facts based on results; 
        you do not hallucinate, you will try as hard as possible to gather facts & data to back up and prove the research.

        Please make sure you complete the objective above with the following rules:
        1/ You shall do enough research to gather as much information as possible about the objective.
        2/ If there are urls of relevant links & articles, you will scrape it to gather more information.
        3/ After scraping & search, you shall think "Are there any new information I should search & scraping based on the data I collected to increase the research quality?" If answer is yes, continue; But do not do this more than 5 iterations.
        4/ You shall not hallucinate, you shall only write facts & data that you have gathered.
        5/ In the final output, You shall include citations of all reference data & links to back up and prove your research; You shall include citations of all reference data & links to back up and prove your research.
        6/ In the final output, You shall include citations of all reference data & links to back up and prove your research; You shall include citations of all reference data & links to back up and prove your research."""
    )

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs= agent_kwargs,
    memory= memory,
)

def run_agent(query):
    total = 0.0
    with get_openai_callback() as cb:
        result = agent({"input": query})
        total += cb.total_cost
    
    return result, total

#4. Use streamlit to create a web app
def main():
    
    st.set_page_config(page_title="AI Research Agent", page_icon=":robot_face:")

    with st.sidebar:
        st.title(":robot_face: AI Research Agent")
        st.write("As an AI Research Agent I will try my very best to answer to all of your queries!")
        st.write(":man-bowing: Note* this AI agent uses OpenAI gpt-3.5-turbo-16k-0613 LLM, therefore queries are limited to 16,384 tokens.")
        
        st.divider()
        
        db = cluster["user_location"]
        collection = db["location"]

        latitude_array = []
        longitude_array = []
    
        if st.checkbox("See people using our Research Agents"):
            this_location = location()
            try:
                latitude = this_location["coords"]["latitude"]
                longitude = this_location["coords"]["longitude"]
                collection.insert_one({"latitude" : latitude, "longitude" : longitude})
            except TypeError as f:
                st.info(":rotating_light: Couldn't get your location, try allowing permissions for accessing location :rotating_light:")
            finally:
                st.subheader("Users currently using our Research Agents:")
                database_data = collection.find({})
                for this_data in database_data:
                    latitude_array.append(this_data["latitude"])
                    longitude_array.append(this_data["longitude"])
                location_dict = {"col1": latitude_array, "col2": longitude_array, "col3": 5, "col4" : np.random.rand(len(latitude_array), 4).tolist()}
                df1 = pd.DataFrame(location_dict)
                st.map(df1, latitude = "col1", longitude = "col2", size = "col3", color = "col4")
    
        st.divider()
    
        st.caption("Made with :white_heart: by Caleb Kan")
        st.caption(":email: email: calebkan1106@gmail.com")
        st.caption(":bird: twitter: @calebkan_")
    
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you with your research?", "message_cost": 0.0}]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) 
            
            try:
                st.info(message["twitter_message"])
            except:
                pass
            
            try:
                this_cost = message["message_cost"]
                cost_message = f"This response costed: ${this_cost} USD"
                st.code(cost_message, language = 'python')
            except:
                pass
            
    # React to user input
    if query := st.chat_input("Send a message"):
        
        # Display user message in chat message container
        st.chat_message("user").markdown(query)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        result, this_cost = run_agent(query)

        # Display assistant response in chat message container
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            for i in range(0, len(result["output"])):
                full_response += result["output"][i]
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(result["output"])
            
            tweet_array = split_string(str(result['output']), 280)

            count = 0
            for i in range(0, len(tweet_array)):
                try:
                    client.create_tweet(text = tweet_array[i])
                    count += 1
                except:
                    twitter_message = ":rotating_light: Twitter tweets has reached the rate limit. Please wait for available quota to view full response on twitter. :rotating_light:"
                    st.info(twitter_message)
                    break
        
            if count == len(tweet_array):
                twitter_message = ":bird: View full response on twitter via @calebkan_"
                st.info(twitter_message)

            cost_message = f"This response costed: ${this_cost} USD"
            st.code(cost_message, language = 'python')
               
        st.session_state.messages.append({"role": "assistant", "content": result["output"], "twitter_message": twitter_message, "message_cost": this_cost})
        
    file_chain = ""
    total = 0.0
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            try:
                total += float(message["message_cost"])
            except:
                pass
            finally:
                file_chain += (str(message["content"]) + "\n")
    
    cost = f'''Total accumulated response cost: ${total} USD'''
    
    st.code(cost, language = 'python')
    st.download_button('Download Chat Response(s) as TXT', file_chain)

if __name__ == "__main__":
    main()
    
app = FastAPI()