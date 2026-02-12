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
import sqlalchemy
from sqlalchemy import create_engine, text
import bcrypt

# --- 1. CONFIG & SETUP ---
st.set_page_config(page_title="FiscalMind Pro", page_icon="🏢", layout="wide")
load_dotenv()

# Database Setup (Postgres)
# Use the URL from docker-compose, or default to localhost for testing
DB_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/fiscalmind")
engine = create_engine(DB_URL)

# RAG Setup (Chroma)
# We use a persistent path so data survives restarts
CHROMA_PATH = "/app/chroma_db" if os.path.exists("/app") else "./chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# --- 2. AUTHENTICATION FUNCTIONS ---

def init_db():
    """Creates the Users table if it doesn't exist."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
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
        return False # Username likely taken

def login_user(username, password):
    """Checks credentials and returns User ID."""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT id, password_hash FROM users WHERE username = :u"), {"u": username}).fetchone()
        
    if result and bcrypt.checkpw(password.encode('utf-8'), result[1].encode('utf-8')):
        return result[0] # Return User ID
    return None

# --- 3. RAG FUNCTIONS (MULTI-USER) ---

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

def process_pdf(uploaded_file, user_id):
    """
    Reads PDF -> Chunks -> Embeds -> Stores in User's Private Collection
    """
    # Create a unique collection for this user (e.g., "user_collection_5")
    collection_name = f"user_collection_{user_id}"
    
    # Delete old collection to keep it fresh (Simple version)
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
    
    ids = [str(i) for i in range(len(chunks))]
    embeddings = embedding_model.encode(chunks).tolist()
    
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return len(chunks)

def query_vector_db(query, user_id):
    """Searches ONLY the specific user's collection"""
    collection_name = f"user_collection_{user_id}"
    try:
        collection = chroma_client.get_collection(name=collection_name)
        results = collection.query(
            query_embeddings=embedding_model.encode([query]).tolist(),
            n_results=3
        )
        return results['documents'][0]
    except:
        return [] # No collection found for user

# --- 4. THE UI (LOGIN vs DASHBOARD) ---

# Initialize DB
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
    st.rerun()

st.title("🧠 FiscalMind: Multi-User Workspace")

# API Key Check
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Server Config Error: Missing API Key")
    st.stop()
client = Groq(api_key=api_key)

# ... (Insert Tools & Chat Logic Here - Keeping it brief for readability) ...
# Copy the Tools (get_stock_price, etc.) from the previous version here
# ...

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful financial assistant."}]

# Sidebar Upload
with st.sidebar:
    st.header("My Knowledge Base")
    uploaded_file = st.file_uploader("Upload Private PDF", type=["pdf"])
    if uploaded_file and "file_processed" not in st.session_state:
        with st.status("🔒 Processing Securely...") as status:
            # PASS USER ID TO PROCESS FUNCTION
            num = process_pdf(uploaded_file, st.session_state.user_id)
            st.session_state.file_processed = True
            status.update(label=f"✅ Indexed {num} chunks to your private vault!", state="complete")

# Chat Logic
if prompt := st.chat_input("Ask about your document..."):
    # RETRIEVE FROM PRIVATE COLLECTION
    context_chunks = query_vector_db(prompt, st.session_state.user_id)
    context_text = "\n\n".join(context_chunks)
    
    full_prompt = f"Context (User's Doc): {context_text}\n\nQuestion: {prompt}"
    
    # Display User Message
    st.chat_message("user").write(prompt)
    
    # Display Assistant Response
    with st.chat_message("assistant"):
        # 1. Get the raw stream from Groq
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Answer using the context provided."},
                {"role": "user", "content": full_prompt}
            ],
            stream=True
        )
        
        # 2. THE FIX: Create a generator to strip the JSON and yield only text
        def stream_data():
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        # 3. Write the clean text stream
        response = st.write_stream(stream_data)
        
    # Optional: Save history to session state if you want it to persist across reruns
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})