FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PORT=8080
ENV FLASK_ENV=production

# Expose port
EXPOSE 8080

# Run using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "3", "app:app"]
