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

# 3. Copy only the dependency file
COPY requirements.txt .

# 4. Install the rest of the libraries
# Note: Pip will see 'torch' is already installed and skip it, saving time.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]