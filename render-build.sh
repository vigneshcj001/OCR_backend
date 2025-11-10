#!/usr/bin/env bash
set -e

# Render automatically runs apt installs if apt.txt is present
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Build completed successfully."
