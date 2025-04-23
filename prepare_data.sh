#!/bin/bash
set -e

# Load environment variables from .env

# Check if KAGGLE_USERNAME and KAGGLE_KEY are already set in the environment
if [[ -z "$KAGGLE_USERNAME" || -z "$KAGGLE_KEY" ]]; then
  echo "🔍 Loading from .env since environment variables not found..."
  if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
  else
    echo "❌ .env file not found and KAGGLE_* variables are not set!"
    exit 1
  fi
fi

kaggle datasets download kvpratama/pokemon-images-dataset -p ./data --unzip

echo "✅ Dataset downloaded to ./data"