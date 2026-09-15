from dotenv import load_dotenv
from langchain.agents import create_agent
from langfuse.langchain import CallbackHandler # Different for other SDKs
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()

OPEN_AI_MODEL="gpt-4o-mini"
SYSTEM_PROMPT="You are a helpful assistant and provide answer to user query"

langfuse_handler = CallbackHandler()
tavily_client = TavilyClient()


# create agent with langchain
def main():
    new_agent = create_agent(
        model=OPEN_AI_MODEL,
        system_prompt=SYSTEM_PROMPT,
        tools=[search]
    )
    agent_response = new_agent.invoke(
        {
            "messages": [{
                "role": "user",
                "content": "Tell me the capital of USA"
            }]
        }, config = {
            "callbacks": [langfuse_handler]
        })
    print(f"Agent Response: {agent_response}")


# create tool without langchain (just a method)
def search(query: str) -> str:
    """This tool will take query and respond for that query"""
    return f"This is a search result for {query}"

# create tool with langchain decorator @tool
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