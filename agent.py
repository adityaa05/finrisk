import os
import json
import yfinance as yf  # stock data from yahoo finance
from google.genai.types import GenerateContentConfig
from google.genai import types
import chromadb
from chromadb.utils import embedding_functions

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


def search_knowledge_base(query):
    """Searches the financial knowledge base (Tata Motors Annual Report) for relevant information."""
    try:
        # Connect to the ChromaDB database
        chroma_client = chromadb.PersistentClient(path="./tata_knowledge_base")

        # Use the same embedding function as ingest.py
        sentence_transformer_embedding = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )

        # Get the collection
        collection = chroma_client.get_collection(
            name="financial_knowledge_base",
            embedding_function=sentence_transformer_embedding,
        )

        # Search for relevant chunks
        results = collection.query(
            query_texts=[query], n_results=2  # Get top 3 most relevant chunks
        )

        # Format results for the AI
        if results["documents"] and results["documents"][0]:
            formatted_results = "\n\n---\n\n".join(results["documents"][0])
            return json.dumps(
                {"results": formatted_results, "count": len(results["documents"][0])}
            )
        else:
            return json.dumps({"results": "No relevant information found.", "count": 0})
    except Exception as e:
        return json.dumps({"error": str(e)})


# Function mapping
function_map = {
    "get_stock_prices": get_stock_prices,
    "calculate_diff": calculate_diff,
    "search_knowledge_base": search_knowledge_base,
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
    types.FunctionDeclaration(
        name="search_knowledge_base",
        description="Search the Tata Motors Annual Report knowledge base for financial information like revenue, assets, liabilities, growth metrics, CEO details, company performance, etc. Use this when users ask about Tata Motors financial data or annual report information.",
        parameters=types.Schema(
            type="object",
            properties={
                "query": types.Schema(
                    type="string",
                    description="The search query to find relevant information in the financial knowledge base",
                ),
            },
            required=["query"],
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
            "You are a financial analyst. For Tata Motors questions, always use search_knowledge_base. Cite page numbers."
        ),
        tools=tools,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="ANY")
        ),
        temperature=0.2,
        top_k=40,
        top_p=0.8,
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

    chat._curated_history = []  # Reset the chat history

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
    # Example 1: Stock prices and calculations
    # init_agent(
    #     "What is the current price of IBM and Nvidia? I want to sell at 50 then what is the diff?"
    # )

    # Example 2: Knowledge base search (Tata Motors Annual Report)
    init_agent("What was Tata Motors revenue in 2024?")

"""client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words",
)

print(response.text)
"""
