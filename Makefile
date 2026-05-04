.PHONY: install models benchmark evaluate score plots dashboard test clean

# ── Setup ──────────────────────────────────────────────────────────────────────
install:
	python -m pip install --prefer-binary -r requirements.txt

models:
	bash scripts/install_models.sh

# ── Pipeline ───────────────────────────────────────────────────────────────────
benchmark:
	python -m scripts.run_benchmark

evaluate:
	python -m scripts.evaluate_results

score:
	python -m scripts.quality_score

plots:
	python -m analysis.visualize_results

dashboard:
	streamlit run dashboard/app.py

# Run full pipeline end-to-end
all: benchmark evaluate score plots

# ── Tests ──────────────────────────────────────────────────────────────────────
test:
	python -m pytest tests/ -v

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf analysis/output/
