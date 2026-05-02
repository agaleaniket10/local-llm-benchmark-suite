# Architecture

## Overview

The benchmark suite is a lightweight Python pipeline that runs prompts against local LLMs via Ollama, records timing and output data, and produces analysis plots.

```
prompts.json
     │
     ▼
run_benchmark.py  ──►  Ollama (local inference)
     │                    ├── mistral:7b-instruct
     │                    ├── llama3.1:8b-instruct
     │                    └── gemma2:9b
     ▼
results.csv
     │
     ├──► evaluate_results.py  ──►  console summary
     │
     └──► visualize_results.py ──►  PNG plots
               │
               └──► (optional) dashboard/app.py  ──►  Streamlit UI
```

## Components

### `scripts/run_benchmark.py`
The main entry point. Iterates over every model × prompt × run combination, calls Ollama's Python SDK, and writes rows to `data/results.csv`.

### `scripts/evaluate_results.py`
Reads `results.csv` and prints per-model, per-category summary statistics (mean latency, p95 latency, tokens/second).

### `scripts/utils.py`
Shared helpers: config loading, prompt loading, timestamp generation, directory creation.

### `models/model_config.py`
Typed dataclass registry of all models with metadata (parameter count, context length, notes).

### `analysis/latency_plots.py`
Matplotlib functions for latency bar charts, grouped category charts, and tokens/second box plots.

### `analysis/visualize_results.py`
Orchestrates all plot generation from a single entry point.

### `dashboard/app.py`
Optional Streamlit app for interactive exploration of results with filters and charts.

## Data Flow

1. `config.yaml` drives all paths and model lists — no hardcoded values in scripts.
2. `data/prompts.json` defines the prompt corpus (id, category, text).
3. `data/results.csv` is the single source of truth for all downstream analysis.
4. Analysis outputs go to `analysis/output/` (gitignored).
