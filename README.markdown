SEE HOW IT WORKS ON: https://calebkan.streamlit.app/

AI Research Agent: A Revolution in Information Gathering

The AI Research Agent is a cutting-edge technology that revolutionizes
the way we research and gather information. This highly innovative tool
incorporates artificial intelligence and web scraping techniques to
facilitate comprehensive, efficient, and reliable data acquisition on
any topic.

Key Functions:

Data Fetching: The AI Research Agent navigates the internet,
meticulously searching through a broad array of sources, including
academic articles, blogs, news sites, and databases. It's programmed to
bypass irrelevant information and zero in on the most valuable,
reliable, and pertinent content related to the selected topic.

Data Summarization: One of the most time-consuming aspects of research
is distilling vast amounts of information into a concise, understandable
summary. The AI Research Agent's sophisticated algorithms process the
fetched data and generate brief, accurate summaries, eliminating the
need for manual sorting and synthesis of information.

Citations: To ensure the credibility of the information provided, the AI
Research Agent automatically includes citations for every piece of data
it collects. It's capable of generating references in various styles,
including APA, MLA, and Chicago, among others.

Advantages:

Efficiency: The AI Research Agent significantly reduces the time spent
on research. It delivers the most relevant information in a fraction of
the time it would take a human researcher, freeing up time for more
in-depth analysis or other tasks.

Accuracy: It uses advanced algorithms to ensure the accuracy of the data
it collects. It's designed to minimize errors and biases, providing the
user with the most reliable information.

Versatility: The AI Research Agent can be utilized across a wide range
of disciplines, from business and economics to science and humanities.
It has the potential to drastically change the way we conduct research
in many fields.

User-friendly: Its interface is intuitive and straightforward, making it
easy for users of all levels to navigate and use effectively.

Ethical Web Scraping: The agent adheres to all ethical guidelines and
legal norms associated with web scraping. It respects website terms of
service and doesn't overload servers, ensuring its data acquisition
methods are responsible and sustainable.

The AI Research Agent is a powerful tool that has the potential to
transform the future of research across various fields. By making
information gathering more efficient and reliable, it allows researchers
to focus on analysis and interpretation, fostering innovation and
advancements in knowledge.

To run this program, you will need to install all the libaries in
"requirements.txt", you will also need to generate API keys from OpenAI,
Browserless, and Serper. These keys will be used to authenticate your
program and allow it to access the respective services.

Follow these steps to create your API keys and set up your environment:

Create API keys:

OpenAI: Visit the OpenAI API webpage and follow the instructions to
create an API key. Browserless: Navigate to the Browserless API section
and create an API key following the provided instructions. Serper: Visit
the Serper API page and generate an API key as per the guidelines.
Create a .env file: Once you have created your API keys, you need to
store them in a .env file. This is a hidden file that stores environment
variables, which are typically configuration values. You can create this
file in the root directory of your project.

Set up your environment variables: In your .env file, assign your API
keys to the appropriate variable names. Each variable should be on its
own line, in the format VARIABLE_NAME="Your API Key". Here's what it
should look like:

OPENAI_API_KEY="your_openai_api_key"
BROWSERLESS_API_KEY="your_browserless_api_key"
SERP_API_KEY="your_serper_api_key"

Replace "your_openai_api_key", "your_browserless_api_key", and
"your_serper_api_key" with the respective keys you generated earlier.
Make sure to keep the quotation marks to ensure the keys are treated as
strings.

That's it! With these steps, your program should be able to use the keys
to authenticate with the respective services.

Note: Ensure that your .env file is listed in your .gitignore file to
prevent it from being committed to your git repository. This helps to
keep your keys secure.

Finally, to run the app, place the command "streamlit run app.py" in
your code interpreter terminal and the app should run.
