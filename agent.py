import time
import os
import yfinance as yf
import matplotlib.pyplot as plt
from duckduckgo_search import DDGS # <--- The News Reader
import google.generativeai as genai
from dotenv import load_dotenv
from colorama import Fore, Style, init

# 1. SETUP
init(autoreset=True)
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- TOOL 1: STOCK PRICE ---
def get_stock_price(ticker: str):
    """Get real-time price."""
    print(f"{Fore.CYAN}📈 [TOOL] Fetching price for: {ticker}...{Style.RESET_ALL}")
    try:
        stock = yf.Ticker(ticker)
        return {"price": round(stock.fast_info.last_price, 2)}
    except:
        return {"error": "Ticker not found"}

# --- TOOL 2: CHART GENERATOR ---
def generate_stock_chart(ticker: str, period: str = "1mo"):
    """Generates a chart image."""
    filename = f"{ticker}_{period}_chart.png"
    print(f"{Fore.MAGENTA}🎨 [TOOL] Drawing chart for {ticker}...{Style.RESET_ALL}")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty: return {"error": "No data"}
        
        plt.figure(figsize=(10, 5))
        plt.plot(hist.index, hist['Close'], label=f'{ticker} Close')
        plt.title(f'{ticker} - {period}')
        plt.grid(True)
        plt.legend()
        plt.savefig(filename)
        plt.close()
        return {"status": "saved", "file": filename}
    except Exception as e:
        return {"error": str(e)}

# --- TOOL 3: WEB SEARCH (New) ---
def search_web(query: str):
    """
    Searches the internet for news and current events.
    Args:
        query: The search query (e.g. 'Why is Tesla stock down today?').
    """
    print(f"{Fore.YELLOW}🌍 [TOOL] Searching web for: {query}...{Style.RESET_ALL}")
    try:
        results = DDGS().text(query, max_results=3)
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}

# 2. REGISTER TOOLS
tools_list = [get_stock_price, generate_stock_chart, search_web]

# 3. INITIALIZE MODEL
model = genai.GenerativeModel(
    model_name='gemini-2.0-flash-001',
    tools=tools_list,
    system_instruction="You are a Senior Investment Analyst. You combine Data (Price), Visuals (Charts), and Context (News) to give a complete answer. If asked 'Why', ALWAYS search for news."
)

# 4. START CHAT
chat = model.start_chat(enable_automatic_function_calling=True)

print(f"{Fore.GREEN}🤖 Super-Analyst Online. (Data + Visuals + News){Style.RESET_ALL}")

while True:
    user_input = input(f"{Fore.BLUE}Manager: {Style.RESET_ALL}")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    try:
        response = chat.send_message(user_input)
        print(f"{Fore.GREEN}Analyst: {response.text}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

# 5. THE ROBUST CHAT LOOP
print(f"{Fore.GREEN}🤖 Agent V3 Online. (Auto-Retry Enabled){Style.RESET_ALL}")

def send_message_with_retry(chat, message, max_retries=3):
    """
    Sends a message to Gemini. If it hits a rate limit, it waits and retries.
    """
    for attempt in range(max_retries):
        try:
            # Try to send the message
            response = chat.send_message(message)
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                wait_time = (attempt + 1) * 20 # Wait 20s, then 40s, then 60s
                print(f"{Fore.RED}⚠️  Rate Limit Hit. Cooling down for {wait_time} seconds...{Style.RESET_ALL}")
                time.sleep(wait_time)
            else:
                # If it's a real error (not a rate limit), crash immediately
                raise e
    return "❌ Error: Failed after multiple retries. Please try again later."

while True:
    user_input = input(f"{Fore.BLUE}Manager: {Style.RESET_ALL}")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    # Use the new safe function
    try:
        response_text = send_message_with_retry(chat, user_input)
        print(f"{Fore.GREEN}Analyst: {response_text}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Critical Error: {e}{Style.RESET_ALL}")