import os
import json
import yfinance as yf
import matplotlib.pyplot as plt
from duckduckgo_search import DDGS
from groq import Groq # <--- The Speed Demon
from dotenv import load_dotenv
from colorama import Fore, Style, init

# 1. SETUP
init(autoreset=True)
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- TOOLS (Same logic, but returning JSON strings) ---
def get_stock_price(ticker: str):
    print(f"{Fore.CYAN}📈 [TOOL] Fetching price for: {ticker}...{Style.RESET_ALL}")
    try:
        stock = yf.Ticker(ticker)
        price = stock.fast_info.last_price
        return json.dumps({"price": round(price, 2), "currency": stock.fast_info.currency})
    except:
        return json.dumps({"error": "Ticker not found"})

def generate_stock_chart(ticker: str, period: str = "1mo"):
    filename = f"{ticker}_{period}_chart.png"
    print(f"{Fore.MAGENTA}🎨 [TOOL] Drawing chart for {ticker}...{Style.RESET_ALL}")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty: return json.dumps({"error": "No data"})
        
        plt.figure(figsize=(10, 5))
        plt.plot(hist.index, hist['Close'], label=f'{ticker} Close')
        plt.title(f'{ticker} - {period}')
        plt.grid(True)
        plt.legend()
        plt.savefig(filename)
        plt.close()
        return json.dumps({"status": "saved", "file": filename})
    except Exception as e:
        return json.dumps({"error": str(e)})

def search_web(query: str):
    print(f"{Fore.YELLOW}🌍 [TOOL] Searching web for: {query}...{Style.RESET_ALL}")
    try:
        results = DDGS().text(query, max_results=3)
        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": str(e)})

# --- TOOL DEFINITIONS (The "Menu" for the AI) ---
# This is the industry standard format (JSON Schema)
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get current stock price",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock symbol (e.g. AAPL)"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_stock_chart",
            "description": "Generate and save a stock chart image",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock symbol"},
                    "period": {"type": "string", "description": "Time period (1d, 5d, 1mo, 1y)"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search internet for news",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }
]

# --- THE AGENT LOOP ---
messages = [
    {
        "role": "system", 
        "content": (
            "You are a financial analysis engine. "
            "When you need external information (like prices, charts, or news), "
            "you MUST call the tools directly. "
            "DO NOT write conversational text like 'Let me check' or 'I will search'. "
            "Just trigger the tool function immediately."
        )
    }
]

print(f"{Fore.GREEN}⚡ Groq Agent Online. (Powered by Llama-3-70b){Style.RESET_ALL}")

while True:
    user_input = input(f"{Fore.BLUE}Manager: {Style.RESET_ALL}")
    if user_input.lower() in ["exit", "quit"]:
        break

    # 1. Add User to Memory
    messages.append({"role": "user", "content": user_input})

    # 2. Call the AI (First Pass)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", # The "Smart" Model
        messages=messages,
        tools=tools_schema,
        tool_choice="auto",
        temperature=0
    )

    response_msg = response.choices[0].message
    tool_calls = response_msg.tool_calls

    if tool_calls:
        # 3. AI wants to use tools
        messages.append(response_msg) # Add the "thought" to memory
        
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            # Execute Python Code
            if func_name == "get_stock_price":
                result = get_stock_price(args["ticker"])
            elif func_name == "generate_stock_chart":
                result = generate_stock_chart(args["ticker"], args.get("period", "1mo"))
            elif func_name == "search_web":
                result = search_web(args["query"])
            
            # 4. Feed Result back to AI
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": func_name,
                "content": result
            })

        # 5. Call AI (Second Pass - Final Answer)
        final_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0
        )
        final_text = final_response.choices[0].message.content
        print(f"{Fore.GREEN}Analyst: {final_text}{Style.RESET_ALL}")
        messages.append({"role": "assistant", "content": final_text})
        
    else:
        # No tool needed
        print(f"{Fore.GREEN}Analyst: {response_msg.content}{Style.RESET_ALL}")
        messages.append({"role": "assistant", "content": response_msg.content})