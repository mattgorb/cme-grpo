# Toy CME-GRPO Eval Results

**Date:** 2026-04-17
**Dataset:** MBPP (sanitized test split, 100 samples)

## Model Eval

| Model | Syntax | Tests | Ref CE | Ref PPL | Entropy |
|-------|--------|-------|--------|---------|---------|
| bigcode/tiny_starcoder_py | 0.0% | 0.0% | 1.832 | 8.4 | 1.64 |
| Salesforce/codegen-350M-mono | 0.0% | 0.0% | 1.349 | 4.6 | 1.10 |

## LLM Judge (gpt-4o-mini)

| Model | Wins | Win Rate |
|-------|------|----------|
| bigcode/tiny_starcoder_py | 20 | 40.0% |
| Salesforce/codegen-350M-mono | 28 | 56.0% |
| Ties | 2 | 4.0% |

**Total samples:** 50

## Recommendation

- **Generator:** bigcode/tiny_starcoder_py (CE=1.832, PPL=8.4, H=1.64)
- **Verifier:** Salesforce/codegen-350M-mono (CE=1.349, PPL=4.6, H=1.10)
- **CE gap:** 0.483 (verifier assigns lower CE to correct solutions)
- **Judge win rate for verifier:** 56.0%
