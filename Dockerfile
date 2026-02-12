FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU first
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Install other dependencies
RUN pip install --no-cache-dir \
    streamlit \
    groq \
    yfinance \
    matplotlib \
    duckduckgo-search \
    python-dotenv \
    PyPDF2 \
    chromadb \
    sentence-transformers

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
