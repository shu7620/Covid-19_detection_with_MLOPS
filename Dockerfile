# Use an official lightweight Python runtime that matches your local environment
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install modern system dependencies required for graphics rendering and AWS CLI
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    awscli \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker's caching mechanism
COPY requirements.txt .

# Strip the local editable install (-e .) from requirements.txt to cache heavy wheels (TensorFlow, etc.)
RUN python -c "with open('requirements.txt', 'r') as f: lines = [l for l in f if '-e' not in l and l.strip() != '.']; open('requirements.txt', 'w').writelines(lines)" && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Now copy the entire project into the container (including pyproject.toml, tests, and src)
COPY . /app

# Install your local cnnClassifier package now that all source files are present
RUN pip install --no-cache-dir .

# Expose the exact port that our Flask app.py is listening on
EXPOSE 8080

# Run the Flask web application server entry point
CMD ["python", "app.py"]