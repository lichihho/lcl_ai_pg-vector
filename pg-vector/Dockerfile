FROM python:3.12.9-slim-bullseye
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY db.py service.py mcp_server.py api_server.py server.py init.sql ./
EXPOSE 8000
ENTRYPOINT ["python", "server.py"]
