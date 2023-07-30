import os

from dotenv import load_dotenv

from streamlit_js_eval import get_geolocation

from pymongo.mongo_client import MongoClient

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

import tweepy

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

uri = "mongodb+srv://caleb:ck110610@cluster.b28bo99.mongodb.net/?retryWrites=true&w=majority"
cluster = MongoClient(uri)

#twitter api set up
#client = tweepy.Client(str(os.getenv("BEAR_TOKEN")), wait_on_rate_limit=True)
client = tweepy.Client(bearer_token = str(os.getenv("BEARER_TOKEN")),
                       consumer_key = str(os.getenv('TWITTER_API_KEY')),
                       consumer_secret = str(os.getenv('TWITTER_API_KEY_SECRET')),
                       access_token = str(os.getenv('ACCESS_TOKEN')),
                       access_token_secret = str(os.getenv('ACCESS_TOKEN_SECRET')),
                       wait_on_rate_limit = False)

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
    content="""You are a top world class researcher who works for OpenAI and is cleared to receive any internal information about OpenAI, ChatGPT, GPT-3.5, and GPT-4. You will conduct detailed research on any topic and produce facts based on results; 
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

def twitter_ai(user_id):
    # Use the Cursor object to get the 10 most recent mentions
    tweets = client.get_users_mentions(id = user_id, max_results = 10)

    for tweet in tweets.data:
        result = agent({"input": tweet.text})
        tweet_array = split_string(str(result["output"]), 280)
        for i in range(0, len(tweet_array)):
            reply = client.create_tweet(text = tweet_array[i], in_reply_to_tweet_id = tweet.id)

def twitter_bot():
    while True:
    # Call the function to read the mentions
        user_id = '794727586362310657' 
        twitter_ai(user_id)
         
    # Wait for 1800 seconds (30 minutes)
        time.sleep(1800)

def signup():
    st.header("Welcome to :robot_face: AI Research Agent! Sign Up below to enter the app!")
    first_name = st.text_input("First Name:")
    last_name = st.text_input("Last Name:")
    email = st.text_input("Email:")
    password = st.text_input("Password:")

    db = cluster["user_info"]
    collection = db["personal_data"]
    
    email_array = []
    
    database_data = collection.find({})
    for this_data in database_data:
        email_array.append(this_data["email"])
        
    btn1 = st.button("Sign Up")
    if btn1:
        if email not in email_array:
            collection.insert_one({"first_name": first_name, "last_name": last_name, "email": email, "password": password})
            st.info(":white_check_mark: Sign Up Success! Return to Log In page to enter the app!")
            st.session_state.runpage = main_app
            st.session_state.runpage()
            st.experimental_rerun()
        else:
            st.info(":rotating_light: Account already exists, try logging in again. :rotating_light:")
    
    btn2 = st.button("Already have a account? Return to Log In Page!")
    if btn2:
        st.session_state.runpage = login
        st.session_state.runpage()
        st.experimental_rerun()

def login():
    st.header("Welcome to :robot_face: AI Research Agent! Log In below to enter the app!")
    email = st.text_input("Email:")
    password = st.text_input("Password:")
    
    db = cluster["user_info"]
    collection = db["personal_data"]
    
    email_array = []
    password_array = []
    
    database_data = collection.find({})
    for this_data in database_data:
        email_array.append(this_data["email"])
        password_array.append(this_data["password"])
    
    btn1 = st.button("Don't have a account? Return to Sign Up Page!")
    
    btn2 = st.button("Log In")
    
    if btn2:
        if email in email_array and password in password_array:
            st.session_state.runpage = main_app
            st.session_state.runpage()
            st.experimental_rerun()
        else:
            st.info(":rotating_light: Check if the email or password entered is correct :rotating_light:")
    
    if btn1:
        st.session_state.runpage = signup
        st.session_state.runpage()
        st.experimental_rerun()

#4. Use streamlit to create a web app
def main():
    st.set_page_config(page_title="AI Research Agent", page_icon=":robot_face:")

    st.header(":robot_face: AI Research Agent")
    st.write("As an AI Research Agent I will try my very best to answer to all of your queries!")
    st.write(":man-bowing: Note* this AI agent uses OpenAI GPT-3.5-Turbo LLM, therefore the query is limited to 4097 tokens.")
    query = st.text_input("Research Title:")

    if query:
        st.write("Doing research for: ", query)

        result = agent({"input": query})
        
        st.info(result['output'])
        
        tweet_array = split_string(str(result['output']), 280)

        count = 0
        for i in range(0, len(tweet_array)):
            try:
                response = client.create_tweet(text = tweet_array[i])
                count += 1
            except tweepy.errors.TooManyRequests:
                st.info(":rotating_light: Twitter tweets has reached the rate limit. Please wait for available quota to view full response on twitter. :rotating_light:")
                break
        
        if count == len(tweet_array):
            st.info(":bird: View full response on twitter via @calebkan_")
            
        st.download_button('Download as TXT', str(result['output'])) 

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
    

if __name__ == "__main__":
    main()

#twitter_bot()


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