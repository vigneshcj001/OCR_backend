#!/usr/bin/env bash
apt-get update && apt-get install -y libleptonica-dev libtesseract-dev pkg-config \
    libpng-dev libjpeg-dev libtiff-dev zlib1g-dev git automake autoconf libtool \
    make g++ wget

# Build from source (v5.5.1)
cd /tmp
wget https://github.com/tesseract-ocr/tesseract/archive/refs/tags/5.5.1.tar.gz
tar -xvzf 5.5.1.tar.gz
cd tesseract-5.5.1
./autogen.sh
./configure
make
make install
ldconfig

# Install Python dependencies
pip install -r requirements.txt
