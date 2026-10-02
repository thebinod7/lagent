# We need environemnt
# we need langfuse
# we need langchain agents | tools
# we need create tools
# need dynamodb connector with langchain and aws
# middleware | dynamodb | guardrails | memory | context engg
# tavily
# multi agent creation

from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler 
from langchain.agents import create_agent
from langchain.tools import tool
import boto3
from langgraph_checkpoint_aws import DynamoDBSaver
from tavily import TavilyClient
from langchain.agents.middleware import SummarizationMiddleware

load_dotenv()

CHECKPOINTER_TABLE = "checkpointers"
AWS_REGION="eu-north-1"
OPEN_AI_MODEL="gpt-4o-mini"

SYSTEM_PROMPT = """
You are a helpful assistant.

GREETING RULE (applies to EVERY final reply, including replies that follow a tool call):
- Look through the conversation history for the user's name.
- If you find it, start your reply with "Hi <name>,".
- If you don't, start with "Hi Batman,".

Example:
User: What is 2+2?
Assistant: Hi John, 2+2 is 4.
"""


langfuseHandler = CallbackHandler()
tavilyClient = TavilyClient()

checkpointer = DynamoDBSaver(
    table_name=CHECKPOINTER_TABLE,
    region_name=AWS_REGION
)

calculator_agent= create_agent(
    model=OPEN_AI_MODEL,
    system_prompt="You are helpful assitant to do mathmatical calculations"
)

@tool
def web_search(query: str) -> str:
    """This tool helps in web search for a query"""
    research = tavilyClient.search(
        query=query
    )
    return research

@tool
def calculator(query: str) -> str:
    """Solve math problems and calculations. Pass the full problem as plain text."""
    result = calculator_agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result["messages"][-1].content

def main():
    boto3.client(
        "dynamodb",
        region_name=AWS_REGION
    )

    agent = create_agent(
        model=OPEN_AI_MODEL,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        middleware=[SummarizationMiddleware(
            model=OPEN_AI_MODEL,
            trigger={"tokens": 4000}, # Trigger beyond 4000 o/p tokens
            keep=("messages", 20) # keep 20 msg in context, others to memory
        )],
        tools=[web_search, calculator]
    )

    config = {
        "configurable":{
            "thread_id": "john-101"
        },
        "callbacks": [langfuseHandler]
    }
    response = agent.invoke({
        "messages": [{
            "role":"user",
            "content":"What is 4+9"
        }]
    },config=config)

    # response_2 = agent.invoke({
    #     "messages":[{
    #         "role": "user",
    #         "content": "What is my name?"
    #     }]
    # },config=config)

    print(f"agent response: {response['messages'][-1].content}")
    # print(f"agent response_2: {response_2['messages'][-1].content}")


if __name__ == "__main__":
    main()