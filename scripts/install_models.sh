#!/usr/bin/env bash
# install_models.sh — Pull all benchmark models via Ollama

set -e

MODELS=(
  "mistral:7b-instruct"
  "llama3.1:8b-instruct"
  "gemma2:9b"
)

echo "🔍 Checking Ollama is running..."
if ! curl -s http://localhost:11434 > /dev/null; then
  echo "❌ Ollama is not running. Start it with: ollama serve"
  exit 1
fi

echo "📦 Pulling models..."
for model in "${MODELS[@]}"; do
  echo "  → Pulling $model"
  ollama pull "$model"
done

echo "✅ All models installed successfully."
