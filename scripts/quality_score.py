"""
quality_score.py — Use an LLM as a judge to score benchmark responses.

For each response in results.csv, a judge model scores it on 5 dimensions
(accuracy, relevance, clarity, completeness, conciseness) from 1–5.
Scores are saved to data/quality_scores.csv.

Usage:
    python scripts/quality_score.py
    python scripts/quality_score.py --judge gemma2:latest --config config.yaml
"""

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd
import requests

from scripts.utils import get_logger, load_config, load_prompts_as_dict, ensure_dir

logger = get_logger(__name__)

JUDGE_PROMPT = """You are an expert evaluator assessing the quality of AI model responses.

Score the following response on these 5 dimensions, each from 1 to 5:
- accuracy:     1=factually wrong, 3=mostly correct, 5=fully accurate
- relevance:    1=off-topic, 3=partially addresses prompt, 5=directly on-topic
- clarity:      1=confusing, 3=understandable, 5=clear and well-structured
- completeness: 1=missing key info, 3=covers basics, 5=thorough with edge cases
- conciseness:  1=too verbose or too brief, 3=reasonable length, 5=optimal length

PROMPT GIVEN TO MODEL:
{prompt}

MODEL RESPONSE:
{response}

Reply with ONLY a JSON object in this exact format, no explanation:
{{"accuracy": <1-5>, "relevance": <1-5>, "clarity": <1-5>, "completeness": <1-5>, "conciseness": <1-5>}}"""

SCORE_KEYS = ["accuracy", "relevance", "clarity", "completeness", "conciseness"]


def call_judge(
    ollama_host: str, judge_model: str, prompt_text: str, response: str
) -> dict | None:
    """Ask the judge model to score a response. Returns score dict or None on failure."""
    judge_input = JUDGE_PROMPT.format(prompt=prompt_text, response=response)
    try:
        resp = requests.post(
            f"{ollama_host}/api/generate",
            json={"model": judge_model, "prompt": judge_input, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")

        match = re.search(r"\{[^{}]+\}", raw)
        if not match:
            logger.warning("Could not parse JSON from judge response: %.100s", raw)
            return None

        scores = json.loads(match.group())
        for k in SCORE_KEYS:
            if k not in scores:
                logger.warning("Missing key '%s' in judge response", k)
                return None
            if not (1 <= int(scores[k]) <= 5):
                logger.warning("Score out of range for '%s': %s", k, scores[k])
                return None

        return {k: int(scores[k]) for k in SCORE_KEYS}

    except requests.exceptions.Timeout:
        logger.error("Judge call timed out")
        return None
    except Exception as e:
        logger.error("Judge call failed: %s", e)
        return None


def main(config_path: str, judge_model: str) -> None:
    config = load_config(config_path)
    ollama_host = config["ollama"]["host"]
    results_file = config["benchmark"]["results_file"]
    prompts = load_prompts_as_dict(config["benchmark"]["prompt_file"])
    output_file = "data/quality_scores.csv"

    try:
        df = pd.read_csv(results_file)
    except FileNotFoundError:
        logger.error(
            "Results file not found: %s — run run_benchmark.py first.", results_file
        )
        return

    df = df[df["response"].notna() & (df["response"].str.strip() != "")]
    if df.empty:
        logger.warning("No responses found in results.csv.")
        return

    logger.info("Judge model: %s", judge_model)
    logger.info("Scoring %d responses...", len(df))

    ensure_dir(str(Path(output_file).parent))
    fieldnames = ["run_id", "model", "prompt_id", "category"] + SCORE_KEYS + ["total"]

    scored, skipped = 0, 0
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for _, row in df.iterrows():
            prompt_text = prompts.get(row["prompt_id"], "")
            logger.info("Scoring %-10s / %s", row["model"], row["prompt_id"])

            scores = call_judge(ollama_host, judge_model, prompt_text, row["response"])
            if scores:
                total = sum(scores.values())
                writer.writerow(
                    {
                        "run_id": row["run_id"],
                        "model": row["model"],
                        "prompt_id": row["prompt_id"],
                        "category": row["category"],
                        **scores,
                        "total": total,
                    }
                )
                logger.info("  → total=%d/25  %s", total, scores)
                scored += 1
            else:
                skipped += 1

    logger.info("Done: %d scored, %d skipped", scored, skipped)

    scores_df = pd.read_csv(output_file)
    if scores_df.empty:
        logger.warning("No scores were recorded.")
        return

    print("\n" + "=" * 60)
    print("📊 QUALITY SCORE SUMMARY (avg total out of 25)")
    print("=" * 60)
    summary = scores_df.groupby("model")["total"].mean().sort_values(ascending=False)
    for model, avg in summary.items():
        bar = "█" * int(avg)
        print(f"  {model:12s}  {avg:5.2f}  {bar}")

    print("\n📊 BY CATEGORY")
    print("-" * 60)
    cat_summary = scores_df.groupby(["model", "category"])["total"].mean().unstack()
    print(cat_summary.round(1).to_string())
    print(f"\n✅ Full scores saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-judge quality scorer")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--judge",
        default="mistral:latest",
        help="Ollama model tag to use as judge (default: mistral:latest)",
    )
    args = parser.parse_args()
    main(args.config, args.judge)
