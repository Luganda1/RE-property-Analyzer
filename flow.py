import os 
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# template = """Question: {question}

# Answer: Let's think step by step."""
# prompt = ChatPromptTemplate.from_template(template)
# model = OllamaLLM(model="llama3.1")

# # Manually invoke the model with the prompt
# input_data = prompt.format_prompt(question="What is LangChain?").to_string()
# response = model.invoke(input_data)
# print(response)



# Initialize the tool for internet searching capabilities
os.environ["SERPER_API_KEY"] = "a8ac358a26e8c3a9e1ad29da1db9cdd764649717"
search_tool = SerperDevTool()

llm = OllamaLLM(model="llama3.1")

researcher = Agent(
    llm=llm,
    role="Senior Property Acquisition Agent",
    goal="Find fixer wrapper investment properties.",
    backstory="You are a veteran property acquisition agent for a fix and flip company. In this case you're looking for fix wrapper properties around los angeles and neighboring counties to invest in.",
    allow_delegation=False,
    tools=[search_tool],
    verbose=True
)


task1 = Task(
    description= " Search the internet for 20 potential fix-and-flip real estate properties in Los Angeles, California. For each property, conduct an analysis to determine whether it requires moderate or extensive repairs.", 
    expected_output= """
    A detailed report of each of the property .The results should be formatted as shown below: 
    Property Address: 6007 W 75th St, Los Angeles, CA 90045
    Bedrooms: 3bd
    Bathrooms: 2ba
    Square Foots: 1234 sqft
    Asking Price: $1,200,000
    Days one Market: 12 days
    Listing agent: John Deo
    Contact: 818 123 8921
    URL Link: https://www.zillow.com/homes/7803-S-Harvard-Blvd-Los-Angeles,-CA-90047_rb/20938580_zpid/
    Background Information: Nestled in the serene Upper North Westport Heights neighborhood of Westchester, this remodeled single-level 3-bedroom, 1-bathroom ranch is a haven of style and comfort. 
    """,
    agent=researcher,
    output_file="task1_output.txt"
)


crew = Crew(agents=[researcher], tasks=[task1], verbose=True)

task_output = crew.kickoff()
print(task_output)