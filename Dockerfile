FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (Required for Postgres driver)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU (Heavy)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Install Logic + Database libs
RUN pip install --no-cache-dir \
    streamlit \
    groq \
    yfinance \
    matplotlib \
    duckduckgo-search \
    python-dotenv \
    PyPDF2 \
    chromadb \
    sentence-transformers \
    psycopg2-binary \
    sqlalchemy \
    bcrypt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]