# Use lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set environment variable for src layout
ENV PYTHONPATH=/app/src

# Train model during build (optional but useful)
RUN python -m ml_project.train

# Expose Flask port
EXPOSE 5000

# Run API
CMD ["python", "-m", "ml_project.api"]
