#!/usr/bin/env bash
# Run analyze_cme_correctness.py across two generators × three benchmarks.
# Outputs per run: cme_{gen}_{bench}.csv (full rows) + cme_{gen}_{bench}.summary.csv (means/stds)
# Logs in logs/cme_{gen}_{bench}.stdout

set -u
mkdir -p logs

VERIFIER="Qwen/Qwen2.5-Math-7B-Instruct"

run_analyze() {
    local gen_short="$1"
    local gen="$2"
    local bench="$3"
    local dataset="$4"
    local split="$5"
    local pkey="$6"
    local akey="$7"
    local max_samples="$8"

    local out="cme_${gen_short}_${bench}.csv"
    local log="logs/cme_${gen_short}_${bench}.stdout"
    local run_name="cme-${gen_short}-${bench}"

    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $gen_short / $bench"
    echo "  output: $out"
    echo "  log:    $log"
    echo "============================================================"

    python analyze_cme_correctness.py \
        --generator "$gen" \
        --verifier "$VERIFIER" \
        --benchmark "$bench" \
        --benchmark-dataset "$dataset" \
        --benchmark-split "$split" \
        --benchmark-problem-key "$pkey" \
        --benchmark-answer-key "$akey" \
        --max-samples "$max_samples" \
        --output "$out" \
        --wandb-run-name "$run_name" \
        2>&1 | tee "$log"
}

# ───────── Qwen2.5-Math-1.5B ─────────
run_analyze "qwen-math-1.5b" "Qwen/Qwen2.5-Math-1.5B" \
    "math500" "HuggingFaceH4/MATH-500" "test" "problem" "answer" 100

run_analyze "qwen-math-1.5b" "Qwen/Qwen2.5-Math-1.5B" \
    "amc23" "math-ai/amc23" "test" "question" "answer" 40

run_analyze "qwen-math-1.5b" "Qwen/Qwen2.5-Math-1.5B" \
    "aime24" "Maxwell-Jia/AIME_2024" "train" "Problem" "Answer" 30

# ───────── Llama-3.2-1B ─────────
run_analyze "llama-3.2-1b" "meta-llama/Llama-3.2-1B" \
    "math500" "HuggingFaceH4/MATH-500" "test" "problem" "answer" 100

run_analyze "llama-3.2-1b" "meta-llama/Llama-3.2-1B" \
    "amc23" "math-ai/amc23" "test" "question" "answer" 40

run_analyze "llama-3.2-1b" "meta-llama/Llama-3.2-1B" \
    "aime24" "Maxwell-Jia/AIME_2024" "train" "Problem" "Answer" 30



# ───────── Llama-3.2-1B-Instruct ─────────
run_analyze "llama-3.2-1b" "meta-llama/Llama-3.2-1B-Instruct" \
    "math500" "HuggingFaceH4/MATH-500" "test" "problem" "answer" 100

run_analyze "llama-3.2-1b" "meta-llama/Llama-3.2-1B-Instruct" \
    "amc23" "math-ai/amc23" "test" "question" "answer" 40

run_analyze "llama-3.2-1b" "meta-llama/Llama-3.2-1B-Instruct" \
    "aime24" "Maxwell-Jia/AIME_2024" "train" "Problem" "Answer" 30



# ───────── Llama-3.2-1B ─────────
run_analyze "llama-3.2-1b" "meta-llama/Llama-3.2-1B-Instruct" \
    "math500" "HuggingFaceH4/MATH-500" "test" "problem" "answer" 100

run_analyze "llama-3.2-1b" "meta-llama/Llama-3.2-1B-Instruct" \
    "amc23" "math-ai/amc23" "test" "question" "answer" 40

run_analyze "llama-3.2-1b" "meta-llama/Llama-3.2-1B-Instruct" \
    "aime24" "Maxwell-Jia/AIME_2024" "train" "Problem" "Answer" 30

# ───────── Qwen2.5-1.5B (general base — matches Intuitor paper) ─────────
run_analyze "qwen-1.5b" "Qwen/Qwen2.5-1.5B" \
    "math500" "HuggingFaceH4/MATH-500" "test" "problem" "answer" 100

run_analyze "qwen-1.5b" "Qwen/Qwen2.5-1.5B" \
    "amc23" "math-ai/amc23" "test" "question" "answer" 40

run_analyze "qwen-1.5b" "Qwen/Qwen2.5-1.5B" \
    "aime24" "Maxwell-Jia/AIME_2024" "train" "Problem" "Answer" 30

# ───────── Gemma-2-2B (different family) ─────────
run_analyze "gemma-2-2b" "google/gemma-2-2b" \
    "math500" "HuggingFaceH4/MATH-500" "test" "problem" "answer" 100

run_analyze "gemma-2-2b" "google/gemma-2-2b" \
    "amc23" "math-ai/amc23" "test" "question" "answer" 40

run_analyze "gemma-2-2b" "google/gemma-2-2b" \
    "aime24" "Maxwell-Jia/AIME_2024" "train" "Problem" "Answer" 30

echo ""
echo "============================================================"
echo "All runs complete."
echo "============================================================"
ls -1 cme_*_*.summary.csv 2>/dev/null || echo "No summary files found."

echo ""
echo "Merge all summaries into one table:"
echo "  head -1 cme_qwen-math-1.5b_math500.summary.csv > all_runs.csv"
echo "  for f in cme_*_*.summary.csv; do tail -n +2 \"\$f\" >> all_runs.csv; done"
