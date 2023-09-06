# Introducing the AI Research Agent 
In the age of information, having quick and reliable access to accurate data is paramount. The AI Research Agent is here to revolutionize the way we gather information. Powered by advanced artificial intelligence and equipped with web scraping capabilities, this agent is adept at sourcing, summarizing, and citing information on a plethora of topics. Whether you're a researcher in academia, a professional in the industry, or simply a curious mind, the AI Research Agent streamlines the data gathering process, ensuring that you receive high-quality information from diverse sources. Welcome to the future of efficient and comprehensive research.
## Table of Contents  
- [Overview](#overview) 
- [Getting Started](#getting-started) 
	- [Installation](#installation) 
	- [Technical Initialization](#technical-initialization)  	
	- [Functionality](#functionality) 
- [Feedback and Contributions](#feedback-and-contributions) 
## Overview 
AI Research Agent: LLM 💬 + Memory 🧠+ Web Scraping 🕸️ + Tool Use 🧰  
  
In today's fast-paced research environment, efficiency is paramount. With this in mind, I developed an AI Research Agent designed to revolutionize the research process by automating the data extraction from the vast expanse of the internet.  
  
Key Features & Workflow:  
  
1. User Query Initiation: Researchers simply pose a research question to the AI.  
2. Web Search: The AI delves into the web, using the provided question as a search criterion, streamlining the hunt for relevant information.  
3. Relevance Analysis: Once on a site, the AI evaluates the content, determining its relevance to the posed query. This ensures that only pertinent information is extracted.  
4. Web Scraping: Using advanced web scraping techniques, the AI gathers data from the relevant sites.  
5. Information Synthesis: Leveraging the power of a Large Language Model, the agent then processes this data, distilling it into concise, key takeaways.  
6. User Output: The summarized information is then presented to the researcher, providing them with a digestible and relevant answer to their initial query.  
  
The overarching aim of this project was to significantly reduce the time researchers spend sifting through myriad sources. By centralizing and automating this process, the AI Research Agent ensures that researchers can spend more time analyzing and less time searching.
## Getting Started  
### Installation 
To experience the power of the AI Research Agent for yourself, it's incredibly straightforward. All you need to do is [click on this link](https://calebkan.streamlit.app/) and you'll be on your way to a seamless research journey.

### Technical Initialization
- Libaries to install: `tweepy`, `beautifulsoup4`, `bs4`, `fastapi`, `jsonschema`, `jsonschema-specifications`, `langchain`, `langchainplus-sdk`, `matplotlib`, `numpy`, `openai`, `openapi-schema-pydantic`, `pandas`, `pydantic`, `python-dotenv`, `streamlit`, `typer`, `typing_extensions`, `typing-inspect`, `tiktoken`, `streamlit-js-eval`, `pymongo`
- Defining the Large Language Model
	```python
	llm  =  ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
	```

### Functionality
   - **Description**: When the application is in running, it dives into the top search results related to your query. Using a sophisticated large language model, the AI Agent evaluates whether a website contains information relevant to your research objective. If a match is found, the AI Agent extracts the necessary text data. This process is repeated for multiple websites, ensuring a comprehensive search. At the end of this research expedition, the AI Agent crafts a summarized output of its findings, neatly packaged with citations for your reference.
   - **Technical**: 
		- `function to search the web`
			 ```python
		    def  search(query):
				url  =  "https://google.serper.dev/search"
				payload  =  json.dumps({
					"q": query
				})
		
				headers  = {
					'X-API-KEY': serper_api_key,
					'Content-Type': 'application/json'
				}

				 response  =  requests.request("POST", url, headers=headers, data=payload)
				 
				print(response.text)
				
				return  response.text 
			```
			
		- `function to scrape website`
			```python 
			def  scrape_website(objective: str, url: str):
				
				print("Scraping website...")
				
				headers  = {
				'Cache-Control': 'no-cache',
				'Content-Type': 'application/json',
				}

				data  = {
					"url": url
				}

				data_json  =  json.dumps(data)
				post_url  =  f"https://chrome.browserless.io/content?token={browserless_api_key}"
				response  =  requests.post(post_url, headers=headers, data=data_json)

				if  response.status_code  ==  200:
					soup  =  BeautifulSoup(response.content, "html.parser")
					text  =  soup.get_text()
					print("THIS CONTENT:", text)
					if  len(text) >  10000:
						output  =  summary(objective, text)
						return  output
					else:
						return  text

				else:
					print(f"HTTP request failed with status code {response.status_code}")
			
			```
		- `function to produce final summary`
			```python 
			def  summary(objective, content):

				text_splitter  =  RecursiveCharacterTextSplitter(
					separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)

				docs  =  text_splitter.create_documents([content])

				map_prompt  =  """
				Write a summary of the following text for {objective}:
				"{text}"
				SUMMARY:
				"""
				
				map_prompt_template  =  PromptTemplate(
					template=map_prompt, input_variables=["text", "objective"])

				summary_chain  =  load_summarize_chain(
					llm=llm,
					chain_type='map_reduce',
					map_prompt=map_prompt_template,
					combine_prompt=map_prompt_template,
					verbose=True
				)
				
				output  =  summary_chain.run(input_documents=docs, objective=objective)
				
				return  output
			```
		- `creating a research AI agent using the functions above with LangChain & OpenAI`
			```python 
			class  ScrapeWebsiteInput(BaseModel):
				"""Inputs for scrape_website"""
				objective: str  =  Field(
					description="The objective & task that users give to the agent")
				url: str  =  Field(description="The url of the website to be scraped")
			```
			```python 
			class  ScrapeWebsiteTool(BaseTool):
				name  =  "scrape_website"
				description  =  "useful when you need to get data from a website url, pass both the objective and url to the function; DO NOT make up any url, the url shall only be from the search results"
				args_schema: Type[BaseModel] =  ScrapeWebsiteInput

				def  _run(self, objective: str, url: str):
					return  scrape_website(objective, url)
					
				def  _arun(self, objective: str, url: str):
					raise  NotImplementedError("error here")
			```
			```python 
			tools  =  [
					Tool(
					name="Search",
					func=search,
					description="Useful for when you need to answer questions regarding all topics. You shall ask targeted questions."
					),
					ScrapeWebsiteTool(),
				]
			```
			```python
			memory  =  ConversationSummaryBufferMemory(
				memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)
			```
			```python
			system_message  =  SystemMessage(
					content=f"""You are a top world class researcher who works for the United States of America and is cleared to receive any internal information on any topic in every company or organization on Earth. You will conduct extremely detailed research on any topic and produce facts based on results;
					you do not hallucinate, you will try as hard as possible to gather facts & data to back up and prove the research.
					
					Please make sure you complete the objective above with the following rules:
					1/ You shall do enough research to gather as much information as possible about the objective.
					2/ If there are urls of relevant links & articles, you will scrape it to gather more information about the objective.
					3/ After scraping & search, you shall think "Is there any new information I should search & scrape based on the data I collected to increase the research quality?" If the answer is yes, continue; But do not do this more than 5 iterations.
					4/ You shall not hallucinate, you shall only write facts with the data that you have gathered.
					5/ In the final output, You shall include citations of all reference data & links to back up and prove your research; You shall include citations of all reference data & links to back up and prove your research.
					6/ In the final output, You shall include citations of all reference data & links to back up and prove your research; You shall include citations of all reference data & links to back up and prove your research."""
				)
			```
			```python 
			agent_kwargs  = {
				"extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
				"system_message": system_message,
			}
			```
			```python
			global chat_history
			chat_history = []
			```
			```python 
			agent  =  initialize_agent(
				tools,
				llm,
				agent=AgentType.OPENAI_FUNCTIONS,
				verbose=True,
				agent_kwargs=  agent_kwargs,
				memory=  memory,
				chat_history = chat_history
			)
			```
		- `function to return result output & total (query + response) cost`
			```python
			def run_agent(query):
			    total = 0.0
			    with get_openai_callback() as cb:
			        result = agent({"input": query})
			        total += cb.total_cost
			    
			    return result, total
			```
  - **Remember*** to use the command responsibly and always review the summarized content to ensure its relevance and accuracy.
    
## Feedback and Contributions
We value your feedback! If you have any suggestions or have found bugs, please report them to email: calebkan1106@gmail.com.

--- 

Stay tuned for regular updates and new features!
