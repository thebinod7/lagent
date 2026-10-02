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

load_dotenv()

CHECKPOINTER_TABLE = "checkpointers"
AWS_REGION="eu-north-1"
OPEN_AI_MODEL="gpt-4o-mini"


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

@tool
def calculator(expression: str) -> str:
    """I will do calculation on given expression"""

    response = calculator_agent.invoke({
        "messages":[{
            "role": "user",
            "content": expression
        }]
    })

    return response["messages"][-1].content

def main():
    dynamodb = boto3.client(
        "dynamodb",
        region_name=AWS_REGION
    )
    print(f"List of tables in dynamoDB {dynamodb}")

    agent = create_agent(
        model=OPEN_AI_MODEL,
        system_prompt="You are helpful assistant. You help answer user query. Be polite.",
        checkpointer=checkpointer,
        tools=[web_search]
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
            "content":"My name is John and I create AI related videos"
        }]
    },config=config)

    response_2 = agent.invoke({
        "messages":[{
            "role": "user",
            "content": "What is my name?"
        }]
    },config=config)

    print(f"agent response: {response['messages'][-1].content}")
    print(f"agent response_2: {response_2['messages'][-1].content}")


if __name__ == "__main__":
    main()