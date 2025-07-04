FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "run_benchmark.py"]