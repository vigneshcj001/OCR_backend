#!/usr/bin/env bash
set -e

echo "🚀 Starting build process..."

# ==========================================================
# 1️⃣ Install system dependencies
# ==========================================================
echo "📦 Installing Tesseract OCR and required libraries..."
apt-get update
apt-get install -y \
    tesseract-ocr \
    libleptonica-dev \
    libtesseract-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev

# ==========================================================
# 2️⃣ Install Python dependencies
# ==========================================================
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# ==========================================================
# 3️⃣ Verify installation
# ==========================================================
echo "🔍 Checking Tesseract installation..."
tesseract --version

echo "✅ Build completed successfully!"
