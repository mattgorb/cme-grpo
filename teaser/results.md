# CME teaser results — full table

All results from `teaser/cme_<model>_<benchmark>.summary.csv`. 100 problems per dataset, 8 generations per problem.

**Verifier**: `Qwen/Qwen2.5-Math-7B-Instruct` in all runs.

**AUROC** = `P(wrong > correct)` for the given metric: 1.0 = perfect wrong/correct separator, 0.5 = random.

**Bold** = best AUROC within each (benchmark, model) row.

**NaN** means the generator produced 0 correct samples on that benchmark, so no AUROC can be computed.


Metric naming: `H` = predictive entropy `-Σ p log p`; `CE` = cross-entropy `-log p(label)`. `gen` = scored under generator; `ver` = scored under verifier. `full` = whole response; `ans` = `\boxed{}` span only.


## math500

| model (acc) | metric | correct mean ± std (n) | wrong mean ± std (n) | AUROC |
|---|---|---|---|---|
| gemma-2-2b (acc=0.006) | H-gen-full | 1.068 ± 0.529 (n=5) | 1.263 ± 0.491 (n=235) | 0.659 |
|  | H-gen-ans | 0.538 ± 0.728 (n=5) | 1.126 ± 0.738 (n=235) | 0.761 |
|  | H-ver-full | 3.329 ± 3.709 (n=5) | 4.296 ± 1.989 (n=235) | 0.693 |
|  | H-ver-ans | 2.873 ± 4.680 (n=5) | 3.539 ± 2.388 (n=235) | 0.708 |
|  | CE-gen-full | 0.930 ± 0.607 (n=5) | 1.160 ± 0.469 (n=235) | 0.671 |
|  | **CE-gen-ans** | 0.185 ± 0.298 (n=5) | 1.153 ± 1.126 (n=235) | **0.858** |
|  | CE-ver-full | 2.172 ± 1.485 (n=5) | 2.561 ± 1.165 (n=235) | 0.687 |
|  | CE-ver-ans | 1.193 ± 1.625 (n=5) | 2.343 ± 2.051 (n=235) | 0.743 |
| llama-3.2-1b (acc=0.158) | H-gen-full | 0.350 ± 0.112 (n=126) | 0.459 ± 0.158 (n=411) | 0.708 |
|  | H-gen-ans | 0.033 ± 0.118 (n=126) | 0.112 ± 0.343 (n=411) | 0.679 |
|  | H-ver-full | 1.297 ± 0.702 (n=126) | 2.124 ± 1.116 (n=411) | 0.764 |
|  | H-ver-ans | 0.590 ± 0.650 (n=126) | 1.934 ± 1.714 (n=411) | 0.837 |
|  | CE-gen-full | 0.267 ± 0.088 (n=126) | 0.349 ± 0.125 (n=411) | 0.694 |
|  | CE-gen-ans | 0.017 ± 0.061 (n=126) | 0.093 ± 0.383 (n=411) | 0.677 |
|  | CE-ver-full | 0.808 ± 0.222 (n=126) | 0.974 ± 0.369 (n=411) | 0.640 |
|  | **CE-ver-ans** | 0.151 ± 0.335 (n=126) | 0.718 ± 1.001 (n=411) | **0.839** |
| qwen-1.5b (acc=0.030) | H-gen-full | 0.631 ± 0.327 (n=24) | 0.902 ± 0.687 (n=127) | 0.529 |
|  | H-gen-ans | 0.107 ± 0.251 (n=24) | 0.647 ± 0.777 (n=127) | 0.799 |
|  | H-ver-full | 2.400 ± 1.933 (n=24) | 4.485 ± 2.778 (n=127) | 0.890 |
|  | H-ver-ans | 0.623 ± 0.705 (n=24) | 2.953 ± 2.711 (n=127) | 0.909 |
|  | CE-gen-full | 0.536 ± 0.300 (n=24) | 0.768 ± 0.580 (n=127) | 0.518 |
|  | CE-gen-ans | 0.083 ± 0.219 (n=24) | 0.587 ± 0.857 (n=127) | 0.780 |
|  | CE-ver-full | 1.736 ± 1.237 (n=24) | 2.775 ± 2.072 (n=127) | 0.861 |
|  | **CE-ver-ans** | 0.154 ± 0.351 (n=24) | 1.527 ± 1.616 (n=127) | **0.919** |
| qwen-math-1.5b (acc=0.271) | H-gen-full | 0.276 ± 0.190 (n=217) | 0.451 ± 0.420 (n=291) | 0.683 |
|  | H-gen-ans | 0.041 ± 0.186 (n=217) | 0.291 ± 0.643 (n=291) | 0.811 |
|  | H-ver-full | 0.895 ± 0.853 (n=217) | 1.709 ± 1.642 (n=291) | 0.784 |
|  | H-ver-ans | 0.250 ± 0.585 (n=217) | 1.143 ± 1.689 (n=291) | 0.854 |
|  | CE-gen-full | 0.250 ± 0.174 (n=217) | 0.412 ± 0.366 (n=291) | 0.683 |
|  | CE-gen-ans | 0.034 ± 0.240 (n=217) | 0.295 ± 0.801 (n=291) | 0.817 |
|  | CE-ver-full | 0.566 ± 0.402 (n=217) | 0.951 ± 0.928 (n=291) | 0.743 |
|  | **CE-ver-ans** | 0.060 ± 0.197 (n=217) | 0.651 ± 1.340 (n=291) | **0.862** |

## amc23

| model (acc) | metric | correct mean ± std (n) | wrong mean ± std (n) | AUROC |
|---|---|---|---|---|
| gemma-2-2b (acc=0.000) | — all metrics | 0 correct samples | — | NaN (undefined) |
| llama-3.2-1b (acc=0.078) | H-gen-full | 0.393 ± 0.150 (n=25) | 0.588 ± 0.252 (n=207) | 0.718 |
|  | H-gen-ans | 0.151 ± 0.511 (n=25) | 0.183 ± 0.550 (n=207) | 0.659 |
|  | H-ver-full | 1.517 ± 0.928 (n=25) | 2.680 ± 1.317 (n=207) | 0.788 |
|  | H-ver-ans | 0.938 ± 1.295 (n=25) | 2.198 ± 1.550 (n=207) | 0.790 |
|  | CE-gen-full | 0.297 ± 0.113 (n=25) | 0.444 ± 0.191 (n=207) | 0.713 |
|  | CE-gen-ans | 0.081 ± 0.318 (n=25) | 0.114 ± 0.447 (n=207) | 0.662 |
|  | CE-ver-full | 0.798 ± 0.201 (n=25) | 1.115 ± 0.481 (n=207) | 0.719 |
|  | **CE-ver-ans** | 0.373 ± 0.738 (n=25) | 1.070 ± 1.405 (n=207) | **0.810** |
| qwen-1.5b (acc=0.016) | H-gen-full | 0.774 ± 0.480 (n=5) | 0.967 ± 0.621 (n=56) | 0.485 |
|  | H-gen-ans | 0.647 ± 0.836 (n=5) | 0.938 ± 0.951 (n=56) | 0.602 |
|  | H-ver-full | 5.006 ± 2.518 (n=5) | 5.002 ± 2.570 (n=56) | 0.717 |
|  | H-ver-ans | 1.950 ± 1.831 (n=5) | 4.355 ± 3.013 (n=56) | 0.802 |
|  | CE-gen-full | 0.661 ± 0.422 (n=5) | 0.840 ± 0.512 (n=56) | 0.474 |
|  | CE-gen-ans | 0.301 ± 0.425 (n=5) | 1.045 ± 1.314 (n=56) | 0.652 |
|  | CE-ver-full | 3.408 ± 1.985 (n=5) | 2.666 ± 1.606 (n=56) | 0.654 |
|  | **CE-ver-ans** | 0.488 ± 0.569 (n=5) | 2.174 ± 1.761 (n=56) | **0.852** |
| qwen-math-1.5b (acc=0.128) | H-gen-full | 0.252 ± 0.161 (n=41) | 0.516 ± 0.532 (n=150) | 0.711 |
|  | H-gen-ans | 0.175 ± 0.639 (n=41) | 0.457 ± 0.751 (n=150) | 0.802 |
|  | H-ver-full | 0.763 ± 0.547 (n=41) | 1.767 ± 1.473 (n=150) | 0.774 |
|  | **H-ver-ans** | 0.166 ± 0.343 (n=41) | 1.306 ± 1.738 (n=150) | **0.889** |
|  | CE-gen-full | 0.235 ± 0.158 (n=41) | 0.459 ± 0.386 (n=150) | 0.708 |
|  | CE-gen-ans | 0.089 ± 0.367 (n=41) | 0.507 ± 0.970 (n=150) | 0.814 |
|  | CE-ver-full | 0.497 ± 0.284 (n=41) | 1.020 ± 0.925 (n=150) | 0.743 |
|  | CE-ver-ans | 0.093 ± 0.354 (n=41) | 0.915 ± 1.339 (n=150) | 0.878 |

## aime24

| model (acc) | metric | correct mean ± std (n) | wrong mean ± std (n) | AUROC |
|---|---|---|---|---|
| gemma-2-2b (acc=0.000) | — all metrics | 0 correct samples | — | NaN (undefined) |
| llama-3.2-1b (acc=0.004) | H-gen-full | 0.961 ± 0.000 (n=1) | 0.735 ± 0.232 (n=147) | 0.109 |
|  | H-gen-ans | 0.762 ± 0.000 (n=1) | 0.362 ± 0.852 (n=147) | 0.129 |
|  | H-ver-full | 3.125 ± 0.000 (n=1) | 2.880 ± 1.146 (n=147) | 0.347 |
|  | **H-ver-ans** | 1.128 ± 0.000 (n=1) | 1.995 ± 1.271 (n=147) | **0.762** |
|  | CE-gen-full | 0.707 ± 0.000 (n=1) | 0.558 ± 0.182 (n=147) | 0.144 |
|  | CE-gen-ans | 0.836 ± 0.000 (n=1) | 0.246 ± 0.713 (n=147) | 0.082 |
|  | CE-ver-full | 1.547 ± 0.000 (n=1) | 1.265 ± 0.411 (n=147) | 0.169 |
|  | CE-ver-ans | 1.264 ± 0.000 (n=1) | 0.818 ± 0.832 (n=147) | 0.224 |
| qwen-1.5b (acc=0.000) | — all metrics | 0 correct samples | — | NaN (undefined) |
| qwen-math-1.5b (acc=0.025) | H-gen-full | 0.253 ± 0.127 (n=6) | 0.519 ± 0.438 (n=129) | 0.721 |
|  | **H-gen-ans** | 0.004 ± 0.003 (n=6) | 0.396 ± 0.686 (n=129) | **0.893** |
|  | H-ver-full | 0.870 ± 0.489 (n=6) | 1.815 ± 1.409 (n=129) | 0.791 |
|  | H-ver-ans | 0.249 ± 0.300 (n=6) | 1.491 ± 1.793 (n=129) | 0.858 |
|  | CE-gen-full | 0.238 ± 0.111 (n=6) | 0.481 ± 0.397 (n=129) | 0.705 |
|  | CE-gen-ans | 0.000 ± 0.000 (n=6) | 0.332 ± 0.852 (n=129) | 0.890 |
|  | CE-ver-full | 0.490 ± 0.223 (n=6) | 0.995 ± 0.806 (n=129) | 0.752 |
|  | CE-ver-ans | 0.034 ± 0.046 (n=6) | 0.926 ± 1.725 (n=129) | 0.884 |

## Best-metric tally (across evaluable rows)

| best metric | # rows won |
|---|---|
| CE-ver-ans | 5 |
| H-ver-ans | 2 |
| CE-gen-ans | 1 |
| H-gen-ans | 1 |

All winning metrics use the `-ans` variant (answer span only), and all but two use the verifier (`-ver-`).
Confirms the design choices for config4: `answer_only_cme: true` + verifier-side scoring.
