# Model Tradeoffs

A summary of the practical tradeoffs observed when running Mistral 7B, Llama 3.1 8B, and Gemma 2 9B locally.

## Comparison Matrix

| Attribute            | Mistral 7B Instruct | Llama 3.1 8B Instruct | Gemma 2 9B       |
|----------------------|---------------------|-----------------------|------------------|
| Parameters           | 7B                  | 8B                    | 9B               |
| Context window       | 8K tokens           | 128K tokens           | 8K tokens        |
| Typical latency*     | Fast                | Moderate              | Moderate–Slow    |
| Reasoning quality    | Good                | Very good             | Very good        |
| Coding ability       | Good                | Very good             | Good             |
| Instruction-following| Strong              | Strong                | Strong           |
| RAM required (Q4)    | ~5 GB               | ~6 GB                 | ~7 GB            |
| License              | Apache 2.0          | Llama 3 Community     | Gemma ToU        |

*Latency varies significantly by hardware. Values are relative, not absolute.

## Key Observations

### Mistral 7B
- Fastest inference at equivalent hardware due to lower parameter count.
- Excellent for latency-sensitive applications.
- Slightly weaker on long-context tasks due to 8K window.

### Llama 3.1 8B
- Best balance of quality and speed in this tier.
- 128K context window makes it uniquely suited for document-level tasks.
- Meta's instruction tuning is competitive with larger models.

### Gemma 2 9B
- Highest parameter count leads to slightly slower inference.
- Strong reasoning and factual accuracy.
- Google's training data and RLHF approach produces clean, structured outputs.

## When to Choose Each

| Use Case                        | Recommended Model     |
|---------------------------------|-----------------------|
| Low-latency chat / autocomplete | Mistral 7B            |
| Long document summarization     | Llama 3.1 8B          |
| Structured data extraction      | Gemma 2 9B or Llama   |
| Code generation                 | Llama 3.1 8B          |
| Privacy-sensitive workloads     | Any (all run locally) |

## Hardware Considerations

- All models run on CPU but are significantly faster with a GPU (NVIDIA CUDA or Apple Metal).
- Apple Silicon (M1/M2/M3) via Ollama's Metal backend provides near-GPU performance.
- Minimum recommended RAM: 16 GB system RAM for comfortable multi-model benchmarking.
