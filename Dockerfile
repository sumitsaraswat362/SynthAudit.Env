FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Python deps
COPY server/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gradio>=4.0.0

# Copy project
COPY . .

# Expose Gradio port
EXPOSE 7860

# Run Gradio app
CMD ["python", "app.py"]
