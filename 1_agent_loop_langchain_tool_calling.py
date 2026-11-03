from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@tool 
def get_product_price(product_name: str) -> float:
    """
    Lookup the price of a product in catalog and return the price as a float. If the product is not found, return 0.
    """
    print(f"Looking up price for product: {product_name}")
    # Simulate fetching the price from a database or API
    prices = {
        "laptop": "$1299.99",
        "smartphone": "$699.99",
        "headphones": "$199.99",
        "monitor": "$299.99",
        "keyboard": "$99.99",
    }
    return prices.get(product_name.lower(), 0.0)  # Return 0.0 if product not found

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# -- Agent Loop Implementation --
@traceable(
    name="langchanin agent loop",
    project_name="React Under The Hood",
    tags=["agent", "loop", "langchain"],
    metadata={"description": "An agent loop that uses tools to answer questions."}
)
def run_agent_loop(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Starting agent loop for question: {question}")
    print("="*60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {iteration}:")
        ai_response = llm_with_tools.invoke(messages)
        tool_calls = ai_response.tool_calls

        # print("AI RESPONSE:", ai_response)
        # print("CONTENT:", ai_response.content)
        # print("TOOL CALLS:", ai_response.tool_calls)
        # print("ADDITIONAL KWARGS:", ai_response.additional_kwargs)

        if not tool_calls:
            print("No tool calls made. Final response:")
            print(ai_response.content)
            return ai_response.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"Tool call detected: {tool_name} with args {tool_args}")
        tool_to_use = tools_dict.get(tool_name)

        observation = tool_to_use.invoke(tool_args)
        print(f"Observation from tool '{tool_name}': {observation}")

        messages.append(ai_response)
        messages.append(ToolMessage(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            content=str(observation)
        ))


if __name__ == "__main__":
    print("Starting agent loop...")
    run_agent_loop("What is the price of a laptop after applying a gold discount?")
    print("Completed agent loop")