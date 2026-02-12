import streamlit as st
import os
import json
import yfinance as yf
import matplotlib.pyplot as plt
from duckduckgo_search import DDGS
from groq import Groq
from dotenv import load_dotenv
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer

# 1. SETUP PAGE
st.set_page_config(page_title="FiscalMind", page_icon="🧠", layout="wide")
st.title("🧠 FiscalMind: Pro RAG Analyst")

# Load Secrets
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Please set your GROQ_API_KEY in the .env file")
    st.stop()

client = Groq(api_key=api_key)
MODEL = "llama-3.1-8b-instant"

# --- RAG SETUP (The Brain) ---
@st.cache_resource
def load_embedding_model():
    # Downloads a small, fast model (all-MiniLM-L6-v2) to the server
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Initialize Vector DB (Chroma)
# We use a persistent storage so it saves data to disk
chroma_client = chromadb.PersistentClient(path="./chroma_db") 
collection = chroma_client.get_or_create_collection(name="financial_docs")

# 2. SYSTEM PROMPT
SYSTEM_PROMPT = (
    "You are a Senior Financial Analyst. "
    "You have access to tools: get_stock_price, generate_stock_chart, search_web. "
    "You also have access to a document provided by the user. "
    "Use the provided 'Context' to answer questions about the document. "
    "If the answer is not in the Context, say 'I cannot find that in the document'. "
    "Always use tools when asked for real-time market data."
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# 3. HELPER: RAG FUNCTIONS
def process_pdf(uploaded_file):
    """Reads PDF, Chunks it, Embeds it, Stores in DB"""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    
    # Chunking (Split by roughly 500 characters)
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Clear old data (for this demo, we assume 1 document at a time)
    try:
        chroma_client.delete_collection("financial_docs")
        collection = chroma_client.create_collection("financial_docs")
    except:
        pass # Collection might not exist yet
        
    # Embed and Store
    ids = [str(i) for i in range(len(chunks))]
    embeddings = embedding_model.encode(chunks).tolist()
    
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )
    return len(chunks)

def query_vector_db(query):
    """Searches DB for top 3 relevant chunks"""
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3 # Get top 3 matches
    )
    return results['documents'][0]

# 4. DEFINE TOOLS
def get_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        price = stock.fast_info.last_price
        currency = stock.fast_info.currency
        return json.dumps({"price": round(price, 2), "currency": currency})
    except:
        return json.dumps({"error": "Ticker not found"})

def search_web(query):
    try:
        results = DDGS().text(query, max_results=3)
        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": str(e)})

def generate_stock_chart(ticker, period="1mo"):
    filename = f"{ticker}_{period}_chart.png"
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

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get current stock price",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_stock_chart",
            "description": "Create chart",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}, "period": {"type": "string"}}, "required": ["ticker"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search news",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    }
]

# 5. SIDEBAR: PDF UPLOAD
with st.sidebar:
    st.header("Upload Knowledge")
    uploaded_file = st.file_uploader("Upload Annual Report (PDF)", type=["pdf"])
    
    if uploaded_file and "pdf_processed" not in st.session_state:
        with st.status("🧠 Processing Memory...", expanded=True) as status:
            status.write("Reading PDF...")
            num_chunks = process_pdf(uploaded_file)
            status.write(f"Chunking into {num_chunks} pieces...")
            status.write("Embedding vectors...")
            st.session_state.pdf_processed = True
            status.update(label="✅ Knowledge Stored!", state="complete", expanded=False)

# 6. UI: DISPLAY HISTORY
for msg in st.session_state.messages:
    if msg.get("role") == "user":
        st.chat_message("user").write(msg["content"])
    elif msg.get("role") == "assistant" and msg.get("content"):
        st.chat_message("assistant").write(msg["content"])
        if "chart.png" in str(msg["content"]):
            try:
                for file in os.listdir():
                    if file.endswith(".png") and file in msg["content"]:
                        st.image(file)
            except: pass

# 7. MAIN AGENT LOOP
if prompt := st.chat_input("Ask about the PDF or Markets..."):
    
    # --- RAG LOGIC: RETRIEVE CONTEXT ---
    context_text = ""
    if "pdf_processed" in st.session_state:
        # Search DB for relevant chunks
        relevant_chunks = query_vector_db(prompt)
        context_text = "\n\n".join(relevant_chunks)
        
        # Add to message history (Invisible to user, visible to AI)
        rag_prompt = f"Context from Document:\n{context_text}\n\nUser Question: {prompt}"
        st.session_state.messages.append({"role": "user", "content": rag_prompt})
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

    st.chat_message("user").write(prompt)

    # --- EXECUTION ---
    with st.chat_message("assistant"):
        status_container = st.status("🤖 Thinking...", expanded=True)
        
        if context_text:
            status_container.write("📚 Consulting Document Memory...")
        
        for _ in range(5):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=st.session_state.messages,
                    tools=tools_schema,
                    tool_choice="auto",
                    temperature=0
                )
            except Exception as e:
                st.error(f"Error: {e}")
                break

            response_msg = response.choices[0].message
            tool_calls = response_msg.tool_calls

            if tool_calls:
                st.session_state.messages.append({
                    "role": "assistant",
                    "tool_calls": [
                        {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} 
                        for tc in tool_calls
                    ],
                    "content": None
                })
                
                for tool_call in tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    status_container.write(f"⚙️ Running: {func_name}...")
                    
                    result = "Error"
                    if func_name == "get_stock_price": result = get_stock_price(args["ticker"])
                    elif func_name == "search_web": result = search_web(args["query"])
                    elif func_name == "generate_stock_chart": 
                        result = generate_stock_chart(args["ticker"], args.get("period", "1mo"))
                        res_json = json.loads(result)
                        if "file" in res_json: st.image(res_json["file"])
                    
                    st.session_state.messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": result
                    })

            else:
                final_answer = response_msg.content
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                status_container.update(label="✅ Complete", state="complete", expanded=False)
                break