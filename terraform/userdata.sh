#!/bin/bash

# 1. Update & Install Dependencies
sudo apt-get update -y
sudo apt-get install -y git docker.io docker-compose-v2

# 2. Start Docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu

# 3. Clone Your Repository
# (We use public HTTPS clone for simplicity. If private, we need tokens)
cd /home/ubuntu
git clone https://github.com/M-Samad/fiscalmind-app.git app

# 4. Create .env file
cd app
# We will inject these variables via Terraform later
echo "GROQ_API_KEY=${groq_api_key}" >> .env
echo "DATABASE_URL=postgresql://user:password@db:5432/fiscalmind" >> .env
echo "POSTGRES_USER=user" >> .env
echo "POSTGRES_PASSWORD=password" >> .env
echo "POSTGRES_DB=fiscalmind" >> .env

# 5. Start the App
# We use sudo because the 'ubuntu' user might not have picked up group changes yet
sudo docker compose up -d