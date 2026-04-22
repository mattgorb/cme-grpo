# Verifier Separation Results

Generator: Qwen/Qwen2.5-Math-1.5B (base)
Dataset: MATH-500 (50 samples, seed=42, 4 generations each)
Correct solutions: 60/200 (30.0%)

## Results

All AUROCs < 0.5 means correct solutions have *higher* -CME (less surprising to verifier).
Effective AUROC = 1 - AUROC (flipped to match reward = -CME direction).

| Verifier | Family | AUROC | Effective AUROC | Gap | Mean CME (correct) | Mean CME (incorrect) |
|---|---|---|---|---|---|---|
| nvidia/Nemotron-Cascade-8B | Nemotron | 0.3051 | 0.6949 | 1.0837 | -0.6819 | -1.7655 |
| Qwen/Qwen2.5-Math-7B-Instruct | Qwen | 0.2696 | 0.7304 | 0.8761 | -0.4780 | -1.3541 |
| microsoft/Phi-4-mini-reasoning | Phi | 0.4146 | 0.5854 | 0.7310 | -0.8884 | -1.6194 |
| google/gemma-4-E4B-it | Gemma | 0.4054 | 0.5946 | 0.5160 | -0.9377 | -1.4537 |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | Llama | 0.4052 | 0.5948 | 0.3640 | -1.3477 | -1.7117 |
| deepseek-ai/deepseek-math-7b-instruct | DeepSeek | 0.4185 | 0.5815 | 0.3526 | -0.8751 | -1.2277 |

## Recommendation

Best cross-family verifier: **nvidia/Nemotron-Cascade-8B** (highest gap, best effective AUROC among non-Qwen models).

Qwen2.5-Math-7B-Instruct has the best raw separation but shares the Qwen family with the generator, risking reward hacking through shared training data biases.

---

## Baseline Eval: Qwen/Qwen2.5-Math-1.5B (base, no training)

Eval settings: greedy decoding, max_new_tokens=3072, batch_size=2

| Benchmark | pass@1 | Correct / Total |
|---|---|---|
| MATH-500 | 44.0% | 22/50 |
| AMC 2023 | 27.5% | 11/40 |
| AIME 2024 | 6.7% | 2/30 |

TTRL paper reports 7.7% on AIME 2024 for this model (temperature=0.6, 16 samples averaged). Our 6.7% with greedy on 30 problems is consistent.





```
============================================================
Loading Qwen/Qwen2.5-Math-1.5B
============================================================
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 338/338 [00:00<00:00, 949.57it/s]
    [8/100] running acc: 1.000
    [16/100] running acc: 0.562
    [24/100] running acc: 0.500
    [32/100] running acc: 0.469
    [40/100] running acc: 0.500
    [48/100] running acc: 0.479
    [56/100] running acc: 0.500
    [64/100] running acc: 0.469
    [72/100] running acc: 0.458
    [80/100] running acc: 0.438
    [88/100] running acc: 0.455
    [96/100] running acc: 0.448
  math500: 0.4500 (45/100) [1372s]
  >> appended to verifier_candidates.csv

============================================================
SUMMARY
============================================================
Model                               Params  math500      Avg
------------------------------------------------------------
Qwen/Qwen2.5-Math-1.5B                1.5B   45.0%   45.0%


============================================================
Loading Qwen/Qwen2.5-Math-7B
============================================================
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 339/339 [00:01<00:00, 202.35it/s]
    [8/100] running acc: 0.625
    [16/100] running acc: 0.562
    [24/100] running acc: 0.625
    [32/100] running acc: 0.625
    [40/100] running acc: 0.625
    [48/100] running acc: 0.625
    [56/100] running acc: 0.607
    [64/100] running acc: 0.609
    [72/100] running acc: 0.625
    [80/100] running acc: 0.600
    [88/100] running acc: 0.602
    [96/100] running acc: 0.594
  math500: 0.6000 (60/100) [1476s]
  >> appended to verifier_candidates.csv

============================================================
SUMMARY
============================================================
Model                               Params  math500      Avg
------------------------------------------------------------
Qwen/Qwen2.5-Math-7B                  7.6B   60.0%   60.0%




============================================================
Loading Qwen/Qwen2.5-Math-7B-Instruct
============================================================
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 339/339 [00:04<00:00, 69.20it/s]
    [20/100] running acc: 0.750
    [40/100] running acc: 0.725
    [60/100] running acc: 0.750
    [80/100] running acc: 0.713
  math500: 0.7600 (76/100) [564s]
  >> appended to verifier_candidates.csv

============================================================
SUMMARY
============================================================
Model                               Params  math500      Avg
------------------------------------------------------------
Qwen/Qwen2.5-Math-7B-Instruct         7.6B   76.0%   76.0%






============================================================
Loading google/gemma-4-E2B
============================================================
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [01:16<00:00, 25.39it/s]
The following generation flags are not valid and may be ignored: ['top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
    [32/100] running acc: 0.062
    [64/100] running acc: 0.109
    [96/100] running acc: 0.083
  math500: 0.0800 (8/100) [613s]
  >> appended to verifier_candidates.csv

============================================================
SUMMARY
============================================================
Model                               Params  math500      Avg
------------------------------------------------------------
google/gemma-4-E2B                    5.1B    8.0%    8.0%






============================================================
Loading meta-llama/Llama-3.2-1B-Instruct
============================================================
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 146/146 [00:00<00:00, 511.43it/s]
The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
    [8/100] running acc: 0.125
    [16/100] running acc: 0.125
    [24/100] running acc: 0.167
    [32/100] running acc: 0.188
    [40/100] running acc: 0.200
    [48/100] running acc: 0.250
    [56/100] running acc: 0.250
    [64/100] running acc: 0.250
    [72/100] running acc: 0.250
    [80/100] running acc: 0.225
    [88/100] running acc: 0.227
    [96/100] running acc: 0.219
  math500: 0.2400 (24/100) [598s]
  >> appended to verifier_candidates.csv

============================================================
SUMMARY
============================================================
Model                               Params  math500      Avg
------------------------------------------------------------
meta-llama/Llama-3.2-1B-Instruct      1.2B   24.0%   24.0%
((pytorch) ) [ec2-user@ip-10-77-1-204 cme-grpo]$ 
```







"google/gemma-4-E2B"
    [8/100] running acc: 0.000
    [16/100] running acc: 0.062
    [24/100] running acc: 0.042
    [32/100] running acc: 0.031
    [40/100] running acc: 0.050
    [48/100] running acc: 0.104
    [56/100] running acc: 0.089
    [64/100] running acc: 0.078
    [72/100] running acc: 0.083
    [80/100] running acc: 0.075
    [88/100] running acc: 0.068
    [96/100] running acc: 0.062
  math500: pass@1 = 0.0700 (7/100)