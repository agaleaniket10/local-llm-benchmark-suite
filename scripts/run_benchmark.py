"""
run_benchmark.py — Run prompts against each configured model and record results.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --config config.yaml
"""

import argparse
import csv
import json
import sys
import time
import uuid
from pathlib import Path

import requests
import yaml
from tqdm import tqdm


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_prompts(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def run_single(ollama_host: str, model_tag: str, prompt: str) -> dict:
    """Send a single prompt to Ollama and return timing + response data."""
    url = f"{ollama_host}/api/generate"
    start = time.perf_counter()
    resp = requests.post(
        url,
        json={"model": model_tag, "prompt": prompt, "stream": False},
        timeout=120,
    )
    elapsed = time.perf_counter() - start
    data = resp.json()

    tokens = data.get("eval_count", 0)
    return {
        "latency_seconds": round(elapsed, 4),
        "tokens_generated": tokens,
        "tokens_per_second": round(tokens / elapsed, 2) if elapsed > 0 else 0,
        "response": data.get("response", "").replace("\n", " ").strip(),
    }


def main(config_path: str) -> None:
    config = load_config(config_path)
    prompts = load_prompts(config["benchmark"]["prompt_file"])
    results_file = config["benchmark"]["results_file"]
    runs = config["benchmark"]["runs_per_prompt"]
    ollama_host = config["ollama"]["host"]

    Path(results_file).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_id",
        "model",
        "prompt_id",
        "category",
        "latency_seconds",
        "tokens_generated",
        "tokens_per_second",
        "response",
        "timestamp",
    ]

    with open(results_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model in config["models"]:
            print(f"\n🤖 Benchmarking {model['name']} ({model['tag']})")
            for prompt in tqdm(prompts, desc="  Prompts"):
                for run in range(runs):
                    try:
                        result = run_single(ollama_host, model["tag"], prompt["prompt"])
                        writer.writerow(
                            {
                                "run_id": str(uuid.uuid4())[:8],
                                "model": model["name"],
                                "prompt_id": prompt["id"],
                                "category": prompt["category"],
                                **result,
                                "timestamp": timestamp(),
                            }
                        )
                    except Exception as e:
                        print(
                            f"\n  ⚠️  Error on {model['name']} / {prompt['id']} run {run+1}: {e}"
                        )

    print(f"\n✅ Results saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM benchmark")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
