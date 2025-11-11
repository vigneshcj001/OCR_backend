FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y tesseract-ocr libleptonica-dev libtesseract-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# ❗ Use --reload ONLY for development
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
