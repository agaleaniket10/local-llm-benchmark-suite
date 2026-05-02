# Quality Scoring Template

Use this rubric to manually (or semi-automatically) score model responses.

## Scoring Dimensions

Each response is scored 1–5 on the following dimensions:

| Dimension       | 1 (Poor)                          | 3 (Acceptable)                    | 5 (Excellent)                        |
|-----------------|-----------------------------------|-----------------------------------|--------------------------------------|
| **Accuracy**    | Factually wrong or misleading     | Mostly correct, minor errors      | Fully accurate and verifiable        |
| **Relevance**   | Off-topic or ignores the prompt   | Partially addresses the prompt    | Directly and completely on-topic     |
| **Clarity**     | Confusing, hard to follow         | Understandable with some effort   | Clear, well-structured, easy to read |
| **Completeness**| Missing key information           | Covers the basics                 | Thorough, covers edge cases          |
| **Conciseness** | Excessively verbose or too brief  | Reasonable length                 | Optimal length for the task          |

**Total score = sum of all dimensions (max 25)**

---

## Scoring Sheet

| run_id | model | prompt_id | accuracy | relevance | clarity | completeness | conciseness | total | notes |
|--------|-------|-----------|----------|-----------|---------|--------------|-------------|-------|-------|
|        |       |           |          |           |         |              |             |       |       |

---

## Notes

- Score each run independently before comparing models.
- For coding prompts, also check whether the code runs correctly.
- For creative prompts, weight clarity and relevance more heavily than accuracy.
