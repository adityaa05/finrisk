import os
import json
import yfinance as yf  # stock data from yahoo finance
from google.genai.types import GenerateContentConfig
from google.genai import types

# from openai import OpenAI
from termcolor import colored
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def get_stock_prices(symbol):
    try:
        ticker = yf.Ticker(symbol)
        price = ticker.fast_info.last_price
        return json.dumps({"symbol": symbol, "price": price})
    except Exception as e:
        return json.dumps({"error": str(e)})


price_function = types.FunctionDeclaration(
    name="get_stock_prices",
    description="Get current stock price for a US stock symbol (e.g., AAPL, TSLA).",
    parameters=types.Schema(
        type="object",
        properties={
            "symbol": types.Schema(
                type="string",
                description="The stock ticker symbol",
            ),
        },
        required=["symbol"],
    ),
)

tools = types.Tool(function_declarations=[price_function])


def init_agent(query):
    system_instruction = "You are a WallStreet financial analyst. Use data to answer."

    print(colored(f"User: {query}", "blue"))

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[tools],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="AUTO")
        ),
        temperature=0
    )

    chat = client.chats.create(
        model="gemini-2.0-flash-lite",
        config=config,
    )

    response = chat.send_message(query)

    # Check if response has candidates and parts
    if not response.candidates or not response.candidates[0].content.parts:
        print(colored(f"Agent: {response.text}", "green"))
        return

    # Check if first part has a function call
    first_part = response.candidates[0].content.parts[0]
    if hasattr(first_part, "function_call") and first_part.function_call:
        print(colored("Agent decided to call a tool...", "yellow"))

        function_responses = []

        for part in response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call:
                function_call = part.function_call
                function_name = function_call.name
                function_args = dict(function_call.args)

                if function_name == "get_stock_prices":
                    print(colored(f"Executing function: {function_name}", "green"))

                    function_response = get_stock_prices(function_args.get("symbol"))

                    # Create proper function response part
                    function_responses.append(
                        types.Part.from_function_response(
                            name=function_name,
                            response={"result": function_response}
                        )
                    )

        # Send all function responses back to the model
        if function_responses:
            final_response = chat.send_message(function_responses)
            print(colored(f"Agent: {final_response.text}", "green"))
    else:
        # No function call, just print the text response
        print(colored(f"Agent: {response.text}", "green"))

if __name__ == "__main__":
    init_agent("What is the current price of Nvidia?")
        
"""client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words",
)

print(response.text)
"""
