#!/bin/bash

# Define the virtual environment directory name
VENV_DIR="venv"

# Check if the virtual environment directory exists
if [ -d "$VENV_DIR" ]; then
  echo "Activating virtual environment"
  source "$VENV_DIR/bin/activate"
  echo "Virtual enviornment activated"
else
  echo "Virtual environment not found. Creating one..."
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  echo "Virtual enviornment activated"
fi
