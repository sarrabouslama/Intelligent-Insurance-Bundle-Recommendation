# Use a slim Python 3.10 image to save space
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies (needed for LightGBM/XGBoost if they require libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and model
COPY src/ ./src/
COPY api/ ./api/
COPY frontend/ ./frontend/
COPY data/ ./data/
COPY model.joblib .

# Expose the ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Command to run the API (You can change this to run the UI or use a shell script to run both)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]