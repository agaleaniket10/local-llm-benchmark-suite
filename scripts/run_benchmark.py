"""
run_benchmark.py — Run prompts against each configured model and record results.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --config config.yaml
"""

import argparse
import csv
import time
import uuid
from pathlib import Path

import requests
from tqdm import tqdm

from scripts.utils import get_logger, load_config, load_prompts, timestamp, ensure_dir

logger = get_logger(__name__)


def run_single(ollama_host: str, model_tag: str, prompt: str, timeout: int) -> dict:
    """Send a single prompt to Ollama and return timing + response data."""
    url = f"{ollama_host}/api/generate"
    start = time.perf_counter()
    resp = requests.post(
        url,
        json={"model": model_tag, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    elapsed = time.perf_counter() - start

    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Ollama error: {data['error']}")

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
    timeout = config["benchmark"]["timeout_seconds"]
    ollama_host = config["ollama"]["host"]

    ensure_dir(str(Path(results_file).parent))

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

    total = len(config["models"]) * len(prompts) * runs
    logger.info(
        "Starting benchmark: %d models × %d prompts × %d runs = %d calls",
        len(config["models"]),
        len(prompts),
        runs,
        total,
    )

    errors = 0
    with open(results_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model in config["models"]:
            logger.info("Benchmarking %s (%s)", model["name"], model["tag"])
            for prompt in tqdm(prompts, desc=f"  {model['name']}"):
                for run in range(runs):
                    try:
                        result = run_single(
                            ollama_host, model["tag"], prompt["prompt"], timeout
                        )
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
                        csvfile.flush()  # write immediately so partial results are saved
                    except requests.exceptions.Timeout:
                        logger.warning(
                            "Timeout on %s / %s run %d",
                            model["name"],
                            prompt["id"],
                            run + 1,
                        )
                        errors += 1
                    except Exception as e:
                        logger.error(
                            "Failed on %s / %s run %d: %s",
                            model["name"],
                            prompt["id"],
                            run + 1,
                            e,
                        )
                        errors += 1

    if errors:
        logger.warning(
            "Benchmark complete with %d error(s). Results saved to %s",
            errors,
            results_file,
        )
    else:
        logger.info("Benchmark complete. Results saved to %s", results_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM benchmark")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
