FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Set default fallback PORT
ENV PORT=8080
EXPOSE 8080

# CRITICAL: Use shell form execution without brackets so ${PORT} expands dynamically
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
