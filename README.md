# 🧠 Local LLM Benchmark Suite

A fully offline benchmarking system for comparing local language models using [Ollama](https://ollama.com). No cloud APIs, no data leaving your machine.

Benchmarks **Mistral 7B**, **Llama 3.1 8B**, and **Gemma 2 9B** across reasoning, coding, summarization, creative writing, and factual recall — measuring latency, throughput, and response quality.

---

## 📊 What It Measures

| Metric | Description |
|---|---|
| Latency | Seconds per response |
| Throughput | Tokens per second |
| Quality | LLM-as-judge scores (1–5) across 5 dimensions |

Quality dimensions: **accuracy**, **relevance**, **clarity**, **completeness**, **conciseness** (max 25 pts)

---

## 🧱 Stack

- [Ollama](https://ollama.com) — local model inference (Metal-accelerated on Apple Silicon)
- Python 3.10+ — benchmarking and analysis scripts
- Pandas + Matplotlib — results processing and plots
- Streamlit — optional interactive dashboard

---

## 🤖 Models Tested

| Model | Parameters | Context | Notes |
|---|---|---|---|
| `mistral:latest` | 7B | 8K | Fast, strong general-purpose |
| `llama3.1:latest` | 8B | 128K | Best balance of quality and speed |
| `gemma2:latest` | 9B | 8K | Strong reasoning, clean outputs |

---

## 🚀 Setup

### 1. Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS (Homebrew)
brew install ollama
```

### 2. Pull models

```bash
ollama serve &   # start Ollama in background
bash scripts/install_models.sh
```

### 3. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ Running the Benchmark

```bash
# Run all prompts against all models
python scripts/run_benchmark.py

# Summarize results in terminal
python scripts/evaluate_results.py

# Score response quality using LLM-as-judge
python scripts/quality_score.py

# Generate latency and quality plots
python analysis/visualize_results.py

# (Optional) Launch interactive dashboard
streamlit run dashboard/app.py
```

---

## 📁 Project Structure

```
local-llm-benchmark-suite/
├── config.yaml                  # Models, paths, benchmark settings
├── requirements.txt
│
├── data/
│   ├── prompts.json             # Benchmark prompt corpus (5 categories)
│   └── results.csv              # Generated — gitignored
│
├── scripts/
│   ├── install_models.sh        # Pull all models via Ollama
│   ├── run_benchmark.py         # Main benchmark runner
│   ├── evaluate_results.py      # Terminal summary statistics
│   ├── quality_score.py         # LLM-as-judge quality scoring
│   └── utils.py                 # Shared helpers
│
├── models/
│   └── model_config.py          # Model metadata registry
│
├── analysis/
│   ├── visualize_results.py     # Orchestrates all plot generation
│   ├── latency_plots.py         # Matplotlib chart functions
│   └── quality_scoring_template.md  # Manual scoring rubric
│
├── dashboard/
│   └── app.py                   # Streamlit dashboard (optional)
│
└── docs/
    ├── architecture.md          # System design and data flow
    └── tradeoffs.md             # Model comparison and use-case guide
```

---

## ⚙️ Configuration

Edit `config.yaml` to change models, prompt file, number of runs, or output paths:

```yaml
benchmark:
  runs_per_prompt: 1     # increase for more stable averages
  timeout_seconds: 120

models:
  - name: mistral
    tag: mistral:latest
```

---

## 🖥️ Hardware Notes

Tested on **Apple M1 Pro (16 GB unified memory)** with Ollama's Metal backend.

- Minimum recommended: 16 GB RAM
- GPU acceleration: automatic on Apple Silicon via Metal, NVIDIA via CUDA
- CPU-only: works but significantly slower

---

## 📄 License

MIT
