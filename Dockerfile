FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Launch uvicorn dynamically bound to Railway's assigned $PORT
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
