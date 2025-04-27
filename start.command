#!/bin/bash
# One-click start script for NADOO-Video on Mac

# Exit on error
set -e

# Ensure we're in the script's directory
cd "$(dirname "$0")"

# Check Python 3
if ! command -v python3 &> /dev/null; then
  echo "Python3 not found. Please install Python 3."
  exit 1
fi

# Check pip3
if ! command -v pip3 &> /dev/null; then
  echo "pip3 not found. Please install pip3."
  exit 1
fi

# Upgrade pip
python3 -m pip install --upgrade pip

# Patch requirements.txt for gradio compatibility
grep -q '^gradio==' requirements.txt && sed -i '' 's/^gradio==.*/gradio/' requirements.txt

# Install dependencies
pip3 install -r requirements.txt

# Run environment tests
echo "Running environment tests..."
python3 test_env.py
if [ $? -ne 0 ]; then
  echo "Test failed. Please check output above."
  exit 1
fi

echo "Starting NADOO-Video GUI..."
python3 demo_gradio.py
