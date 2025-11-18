# ------------ BASE PYTHON IMAGE ------------
FROM python:3.10-slim

# ------------ WORKDIR ------------
WORKDIR /app

# ------------ INSTALL SYSTEM DEPENDENCIES ------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libtesseract-dev \
        libleptonica-dev \
        libgl1 \
        libglib2.0-0 \
        build-essential \
        pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ------------ COPY PROJECT FILES ------------
COPY . .

# ------------ INSTALL PYTHON DEPENDENCIES ------------
RUN pip install --no-cache-dir -r requirements.txt

# ------------ EXPOSE PORT ------------
EXPOSE 8000

# ------------ START FASTAPI SERVER ------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
