#!/usr/bin/env bash
set -e

# Install system dependencies
apt-get update
apt-get install -y tesseract-ocr libleptonica-dev libtesseract-dev

# Verify installation
tesseract --version

# Install Python dependencies
pip install -r requirements.txt
