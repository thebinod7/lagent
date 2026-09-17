from dotenv import load_dotenv
from langchain.agents import create_agent
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from tavily import TavilyClient
from pydantic import BaseModel

load_dotenv()

langfuse_handler = CallbackHandler()
tavily_client = TavilyClient()


class AnswerResponse(BaseModel):
    answer: str
    confidence: float

system_prompt = """You are a helpful assistant, answer all user query.
You have access to the following tools.

<tools>
1. Tavily web search tool: It allows you to web search using Tavily API. You can use this tool to find information on the internet.
To use this tool, you need to provide a query string as input.

The tool will return results from Tavily API, which you can use to answer user query. The search result will include relevant information.
Call the function `taviliy_web_search(query: str) -> dict:` to use Tavily Web search tool. 
</tools>
"""

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

config = {
    "configurable": {"thread_id": 1}, # chat session key. Could be userId for dynamic session
    "callbacks": [langfuse_handler]
}

agent = create_agent(
    model="gpt-4o-mini",
    system_prompt=system_prompt,
    tools=[tavily_web_search],
    response_format=AnswerResponse,
    checkpointer=InMemorySaver() # Save agent states like conversation_history, intermediate_state
)

r1 = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Do you know about Outbid website that made 200K"
    }]
}, config=config)

# r2 = agent.invoke({
#     "messages": [{
#         "role": "user",
#         "content": "Whats my name"
#     }]
# }, config=config)

print(f"Agent Response: {r1['structured_response']}")
