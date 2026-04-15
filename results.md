# CME correctness-separation results

100 MATH-500 problems, 8 samples each (800 total generations).

Generator: `Qwen/Qwen2.5-Math-1.5B`
Verifier:  `Qwen/Qwen2.5-Math-7B`

Bucketing:
- **correct** — extracted `\boxed{}` matches gold.
- **incorrect** — extracted `\boxed{}` present but wrong.
- **none** — no `\boxed{}` (or empty).

## Per-bucket running means (after problem 100)

```
[running after problem 100] correct=213 incorrect=382 none=205
    ppl_gen_full            correct=   1.283 (n=213)  incorrect=   1.528 (n=382)  none= 109.718 (n=205)
    ppl_gen_answer          correct=   1.039 (n=213)  incorrect=   2.177 (n=382)  none=40136.529 (n=99)
    ppl_ver_full            correct=   3.136 (n=213)  incorrect=  11.726 (n=382)  none= 387.603 (n=205)
    ppl_ver_answer          correct=   1.059 (n=213)  incorrect=  28.121 (n=382)  none=18551.130 (n=99)
    entropy_gen_full        correct=   0.259 (n=213)  incorrect=   0.411 (n=382)  none=   0.763 (n=205)
    entropy_gen_answer      correct=   0.053 (n=213)  incorrect=   0.280 (n=382)  none=   0.697 (n=99)
    entropy_ver_full        correct=   0.897 (n=213)  incorrect=   2.096 (n=382)  none=   4.363 (n=205)
    entropy_ver_answer      correct=   0.275 (n=213)  incorrect=   1.912 (n=382)  none=   6.450 (n=99)
```
