import streamlit as st
import os
import time  # Needed for timestamp in memory
import yfinance as yf # Keeping imports even if unused in this snippet
import matplotlib.pyplot as plt
from duckduckgo_search import DDGS
from groq import Groq
from dotenv import load_dotenv
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
import bcrypt

# --- 1. CONFIG & SETUP ---
st.set_page_config(page_title="FiscalMind Pro", page_icon="🏢", layout="wide")
load_dotenv()

# Database Setup (Postgres)
# Use the URL from docker-compose, or default to localhost for testing
DB_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/fiscalmind")
engine = create_engine(DB_URL)

# --- 2. AUTHENTICATION FUNCTIONS ---

def init_db():
    """Creates the Users and Chats tables if they don't exist."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chats (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.commit()

def register_user(username, password):
    """Hashes password and saves new user."""
    pwd_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO users (username, password_hash) VALUES (:u, :p)"), 
                        {"u": username, "p": pwd_hash})
            conn.commit()
        return True
    except:
        return False

def login_user(username, password):
    """Checks credentials and returns User ID."""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT id, password_hash FROM users WHERE username = :u"), {"u": username}).fetchone()
        
    if result and bcrypt.checkpw(password.encode('utf-8'), result[1].encode('utf-8')):
        return result[0]
    return None

def save_message(user_id, role, content):
    """Saves a single message to the database"""
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO chats (user_id, role, content) VALUES (:uid, :r, :c)"),
            {"uid": user_id, "r": role, "c": content}
        )
        conn.commit()

def load_messages(user_id):
    """Loads chat history for a user"""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT role, content FROM chats WHERE user_id = :uid ORDER BY timestamp ASC"),
            {"uid": user_id}
        ).fetchall()
    return [{"role": row[0], "content": row[1]} for row in result]

# --- 3. RAG & VECTOR FUNCTIONS (MULTI-USER) ---

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# FIX: Cache the Chroma Client to prevent "Client has been closed" error
@st.cache_resource
def get_chroma_client():
    path = "/app/chroma_db" if os.path.exists("/app") else "./chroma_db"
    return chromadb.PersistentClient(path=path)

chroma_client = get_chroma_client()

def get_collection(name):
    """Helper to get a collection by name"""
    return chroma_client.get_or_create_collection(name=name)

def process_pdf(uploaded_file, user_id):
    """Reads PDF -> Chunks -> Embeds -> Stores in User's Private Collection"""
    collection_name = f"docs_{user_id}"
    
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass
        
    collection = chroma_client.create_collection(name=collection_name)
    
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    if not chunks:
        return 0

    ids = [str(i) for i in range(len(chunks))]
    embeddings = embedding_model.encode(chunks).tolist()
    
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return len(chunks)

# --- TIER 3 MEMORY FUNCTIONS ---

def store_memory(user_id, text, metadata={'type': 'chat'}):
    """Saves a specific interaction to Episodic Memory."""
    collection = get_collection(f"memory_{user_id}")
    mem_id = str(time.time()) # Uses current time as unique ID
    
    collection.add(
        documents=[text],
        metadatas=[metadata],
        ids=[mem_id]
    )

def query_memories(user_id, query_text, n_results=2):
    """Searches ONLY the user's past conversations."""
    try:
        collection = chroma_client.get_collection(name=f"memory_{user_id}")
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results['documents'][0]
    except:
        return []

def query_documents(user_id, query_text, n_results=3):
    """Searches ONLY the user's uploaded PDFs."""
    try:
        collection = chroma_client.get_collection(name=f"docs_{user_id}")
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results['documents'][0]
    except:
        return []

# --- 4. THE UI (LOGIN vs DASHBOARD) ---

# Initialize DB Tables
init_db()

# Session State for Auth
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None

# A. LOGIN SCREEN
if st.session_state.user_id is None:
    st.title("🔐 FiscalMind Pro: Login")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        l_user = st.text_input("Username", key="l_user")
        l_pass = st.text_input("Password", type="password", key="l_pass")
        if st.button("Login"):
            uid = login_user(l_user, l_pass)
            if uid:
                st.session_state.user_id = uid
                st.session_state.username = l_user
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        r_user = st.text_input("New Username", key="r_user")
        r_pass = st.text_input("New Password", type="password", key="r_pass")
        if st.button("Register"):
            if register_user(r_user, r_pass):
                st.success("Account created! Please login.")
            else:
                st.error("Username already taken.")

    st.stop() # Stop here if not logged in

# B. MAIN APP (ONLY VISIBLE IF LOGGED IN)
st.sidebar.title(f"👤 {st.session_state.username}")
if st.sidebar.button("Logout"):
    st.session_state.user_id = None
    # st.session_state.messages = [] # Clear local session messages on logout
    st.rerun()

st.title("🧠 FiscalMind: Multi-User Workspace")

# API Key Check
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Server Config Error: Missing API Key")
    st.stop()
client = Groq(api_key=api_key)

# --- CHAT INTERFACE ---

# 1. Load Messages from DB (Logic: Only if we haven't loaded them yet for this user)
if "messages" not in st.session_state or "last_user_id" not in st.session_state or st.session_state.get("last_user_id") != st.session_state.user_id:
    st.session_state.messages = []

# Check if we switched users or first load
if "last_user_id" not in st.session_state or st.session_state.last_user_id != st.session_state.user_id:
    db_msgs = load_messages(st.session_state.user_id)
    st.session_state.messages = db_msgs
    st.session_state.last_user_id = st.session_state.user_id

# 2. Display Chat History
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# 3. Sidebar Upload
with st.sidebar:
    st.header("My Knowledge Base")
    uploaded_file = st.file_uploader("Upload Private PDF", type=["pdf"])
    if uploaded_file and "file_processed" not in st.session_state:
        with st.status("🔒 Processing Securely...") as status:
            num = process_pdf(uploaded_file, st.session_state.user_id)
            st.session_state.file_processed = True
            status.update(label=f"✅ Indexed {num} chunks to your private vault!", state="complete")

# 4. CHAT LOGIC (TIER 3 MEMORY)
if prompt := st.chat_input("Ask me anything..."):
    
    # A. Display User Message Immediately
    st.chat_message("user").write(prompt)
    
    # B. Add to Session State & DB
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(st.session_state.user_id, "user", prompt)

    # C. RETRIEVAL (Dual Brain)
    
    # Brain A: Search Documents (Facts)
    doc_results = query_documents(st.session_state.user_id, prompt)
    doc_context = "\n".join(doc_results) if doc_results else "No relevant document info."

    # Brain B: Search Memories (Past Chats)
    mem_results = query_memories(st.session_state.user_id, prompt)
    mem_context = "\n".join(mem_results) if mem_results else "No relevant past memories."

    # D. CONSTRUCT PROMPT
    system_prompt = f"""
    You are an AI assistant.
    
    MEMORY FROM PAST CONVERSATIONS:
    {mem_context}
    
    CONTEXT FROM UPLOADED DOCUMENTS:
    {doc_context}
    
    Answer the user's question using the information above.
    """
    
    # Sliding Window (Recent 5 turns)
    recent_history = st.session_state.messages[-5:] 

    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in recent_history:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add the current prompt (redundant if in recent_history, but safe to add explicitly if logic varies)
    # Note: recent_history already includes the 'prompt' we just appended above. 
    # To be safe and avoid duplication in API call:
    if recent_history[-1]['content'] != prompt:
        api_messages.append({"role": "user", "content": prompt})

    # E. GENERATE RESPONSE
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=api_messages,
            stream=True
        )
        
        # FIX: Defined stream_data helper HERE
        def stream_data():
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        
        # Write stream
        response = st.write_stream(stream_data)

    # F. AUTO-ARCHIVE (Episodic Memory)
    memory_text = f"User asked: {prompt} | AI Answered: {response}"
    store_memory(st.session_state.user_id, memory_text)
    
    # Save Assistant Response to DB & State
    save_message(st.session_state.user_id, "assistant", response)
    st.session_state.messages.append({"role": "assistant", "content": response})