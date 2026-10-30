
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
# from tavily import TavilyClient
from langchain_tavily import TavilySearch


load_dotenv()
# tavily_client = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """Search for job postings using Tavily API"""
#     print(f"Searching for job postings with query: {query}")
#     return tavily_client.search(query=query, limit=3)


llm = ChatOpenAI(model="gpt-5")
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="search for 3 job postings for an principal software engineer in Banglore on linkedin and list their details?"
                )
            ]
        }
    )
    print(result)

if __name__ == "__main__":
    main()