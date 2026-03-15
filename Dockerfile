# Lab 4: Dockerfile for Wine Quality API
# Student: Srinivas Raghav V C
# Roll No: 2022BCS0016

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create artifacts directory
RUN mkdir -p app/artifacts

# Expose port
EXPOSE 8000

# Labels
LABEL maintainer="2022BCS0016 - Srinivas Raghav V C"
LABEL roll_no="2022BCS0016"

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
