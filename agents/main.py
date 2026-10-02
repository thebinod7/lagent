from dotenv import load_dotenv
from langchain.agents import create_agent
from langfuse.langchain import CallbackHandler # Different for other SDKs
from langchain.tools import tool
from tavily import TavilyClient
from pydantic import BaseModel

load_dotenv()

SEARCH_QUERY="What is the goal of Balen Shah visiting UN general assembly?"
OPEN_AI_MODEL="gpt-4o-mini"

SYSTEM_PROMPT="You are a helpful assistant and provide answer to user query"

class AnswerResponse(BaseModel):
    answer: str
    confidence: float

langfuse_handler = CallbackHandler()
tavily_client = TavilyClient()


# create agent with langchain
def main():
    new_agent = create_agent(
        model=OPEN_AI_MODEL,
        system_prompt=SYSTEM_PROMPT,
        tools=[tavily_web_search],
        response_format=AnswerResponse
    )
    agent_response = new_agent.invoke(
        {
            "messages": [{
                "role": "user",
                "content": SEARCH_QUERY
            }]
        }, config = {
            "callbacks": [langfuse_handler]
        })
    print(f"Agent Response: {agent_response['structured_response']}")


# tool without langchain decorator (just a demo tool)
def search(query: str) -> str:
    """This tool will take query and respond for that query"""
    return f"This is a search result for {query}"

# tool with langchain decorator @tool. This is also demo tool
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return f"The weather of {location} is sunny day."

# add tavily search as a tool
@tool
def tavily_web_search(query: str) -> dict:
    """Search the web using Tavily for up-to-date information on a given query."""
    answer = {
        "title": [],
        "response": []
    }
    response = tavily_client.search(
        query=query,
        search_depth="advanced"
    )
    return response

# Run the function if executed directly. NOT on imports
if __name__ == "__main__":
    main()