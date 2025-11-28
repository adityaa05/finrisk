import os
import json
import yfinance as yf  # stock data from yahoo finance
from google.genai.types import GenerateContentConfig
from google.genai import types

# from openai import OpenAI
from termcolor import colored
from google import genai
from dotenv import load_dotenv


# Configuration
load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# Functions for respective tasks
def get_stock_prices(symbol):
    try:
        ticker = yf.Ticker(symbol)
        price = ticker.fast_info.last_price
        return json.dumps({"symbol": symbol, "price": price})
    except Exception as e:
        return json.dumps({"error": str(e)})


def calculate_diff(current_price, target_price):
    try:
        current_price = float(current_price)
        target_price = float(target_price)
        diff = (target_price - current_price) / current_price * 100
        return json.dumps({"diff": round(diff, 2)})
    except Exception as e:
        return json.dumps({"error": str(e)})


# Function mapping
function_map = {
    "get_stock_prices": get_stock_prices,
    "calculate_diff": calculate_diff,
}

# Function declarations used by the Gemini tools API
tool_declarations = [
    types.FunctionDeclaration(
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
    ),
    types.FunctionDeclaration(
        name="calculate_diff",
        description="Calculate the percentage gain/loss from a current price to a target price.",
        parameters=types.Schema(
            type="object",
            properties={
                "current_price": types.Schema(
                    type="number",
                    description="The current stock price",
                ),
                "target_price": types.Schema(
                    type="number",
                    description="The user's desired target price",
                ),
            },
            required=["current_price", "target_price"],
        ),
    ),
]

# List of Tool objects passed to GenerateContentConfig
tools = [types.Tool(function_declarations=tool_declarations)]


# Agent initialization
def init_agent(query):
    print(colored(f"User: {query}", "blue"))
    # Chat config
    config = types.GenerateContentConfig(
        system_instruction=str(
            "You are a WallStreet financial analyst. Use the tools to get data, then provide a complete, detailed answer that includes all relevant numbers and context.",
        ),
        tools=tools,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="AUTO")
        ),
        temperature=1,
        top_k=40,
        top_p=0.95,
        max_output_tokens=256,
    )
    # Chat creation
    chat = client.chats.create(
        model="gemini-2.0-flash-lite",
        config=config,
    )
    # Chat response
    response = chat.send_message(query)

    # We loop while the model keeps asking for tool calls
    while response.candidates and response.candidates[0].content.parts:

        # 1. Filter the parts to find ONLY function calls
        function_calls = [
            part.function_call
            for part in response.candidates[0].content.parts
            if part.function_call
        ]

        # 2. If no function calls, extract and print ALL text parts
        if not function_calls:
            # Collect all text from parts
            text_parts = []
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)

            final_text = "".join(text_parts).strip()

            if final_text:
                print(colored(f"Agent: {final_text}", "green"))
            else:
                # Fallback to response.text if parts are empty
                print(colored(f"Agent: {response.text}", "green"))
            break

        # 3. If there ARE calls, execute them
        function_responses = []

        for call in function_calls:
            fn_name = call.name
            fn_args = call.args

            # Execute if valid
            if fn_name in function_map:
                try:
                    # **fn_args unpacks the arguments automatically
                    result = function_map[fn_name](**fn_args)
                except Exception as e:
                    result = {"error": f"Function execution failed: {e}"}
            else:
                result = {"error": "Unknown function"}

            # Package the result for Gemini
            function_responses.append(
                types.Part.from_function_response(
                    name=fn_name, response={"result": result}
                )
            )

        # 4. Send ALL results back to the model at once
        response = chat.send_message(function_responses)

    """# Check if response has candidates and parts
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
                            name=function_name, response={"result": function_response}
                        )
                    )

        # Send all function responses back to the model
        if function_responses:
            final_response = chat.send_message(function_responses)
            print(colored(f"Agent: {final_response.text}", "green"))
    else:
        # No function call, just print the text response
        print(colored(f"Agent: {response.text}", "green"))

"""


if __name__ == "__main__":
    init_agent(
        "What is the current price of Nvidia? I want to sell at 50 then what is the diff?"
    )

"""client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words",
)

print(response.text)
"""
