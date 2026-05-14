#!/usr/bin/env bash
# Evaluate the 8 verifiers from the sweep on MATH-500/AMC-23/AIME-24 pass@1.
# Settings match the in-training PeriodicEvalCallback:
#   max_new_tokens=3072, batch_size=2, greedy (temperature=0), max_samples=100 for math500.
# AMC-23 and AIME-24 use their full splits (40 / 30 problems) inside the eval script.
#
# Output: verifier_capability.csv (one row per verifier with pass@1 per benchmark).

set -euo pipefail

mkdir -p logs

VERIFIERS=(
    #"Qwen/Qwen2.5-Math-7B-Instruct"      # v1
    "Qwen/Qwen2.5-Math-1.5B-Instruct"    # v2
    "Qwen/Qwen2.5-Math-1.5B"             # v3
    "Qwen/Qwen2.5-0.5B-Instruct"         # v4
    "Qwen/Qwen2.5-0.5B"                  # v5
    "meta-llama/Llama-3.1-8B-Instruct"   # v6
    "google/gemma-2-2b-it"               # v7
    "allenai/OLMo-2-1124-7B-Instruct"    # v8
)

MODELS_CSV=$(IFS=, ; echo "${VERIFIERS[*]}")

LOG="logs/verifier_capability_$(date +%Y%m%d_%H%M%S).log"
OUT="verifier_capability.csv"

echo "============================================================"
echo "Evaluating ${#VERIFIERS[@]} verifiers on math500/amc23/aime24"
echo "  output: $OUT"
echo "  log:    $LOG"
echo "============================================================"

python eval_verifier_candidates.py \
    --models "$MODELS_CSV" \
    --benchmarks math500,amc23,aime24 \
    --max-new-tokens 3072 \
    --batch-size 2 \
    --max-samples 100 \
    --output "$OUT" \
    2>&1 | tee "$LOG"

echo ""
echo "Done. Results in $OUT"
