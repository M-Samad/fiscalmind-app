import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("🔍 Checking available Groq models...")
try:
    models = client.models.list()
    for m in models.data:
        print(f"✅ ID: {m.id}")
except Exception as e:
    print(f"❌ Error: {e}")