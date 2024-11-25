FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Use PORT environment variable with fallback to 8501
CMD streamlit run --server.port ${PORT:-8501} --server.address 0.0.0.0 new.py