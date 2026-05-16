#!/usr/bin/env bash
# Verifier capability sweep — Qwen2.5-1.5B BASE (NOT the Math variant) as generator.
# 300 training steps, eval every 50, save every 50.
# Hyperparameters otherwise mirror run_verifier_sweep.sh:
# G=8, beta=0, epsilon=0.2 (TRL default), lr=3e-6.

set -euo pipefail

mkdir -p logs configs/sweep_qwen15b

# label : MATH-500 pass@1 (approx) : verifier model
VERIFIERS=(
    "v1:83:Qwen/Qwen2.5-Math-7B-Instruct"
    "v2:75:Qwen/Qwen2.5-Math-1.5B-Instruct"
    "v3:65:Qwen/Qwen2.5-Math-1.5B"
    "v5:20:Qwen/Qwen2.5-0.5B"
    "v6:24:meta-llama/Llama-3.2-1B-Instruct"   # cross-family (Llama)
)

for entry in "${VERIFIERS[@]}"; do
    label="${entry%%:*}"
    rest="${entry#*:}"
    pass="${rest%%:*}"
    verifier="${rest#*:}"
    cfg="configs/sweep_qwen15b/${label}.yaml"

    cat > "$cfg" <<EOF
# Verifier sweep ${label} — generator: Qwen/Qwen2.5-1.5B (base, NOT Math)
# verifier: ${verifier} (~${pass}% MATH-500 pass@1)
model:
  generator: Qwen/Qwen2.5-1.5B
  verifier: ${verifier}

data:
  train_dataset: DigitalLearningGmbH/MATH-lighteval
  max_prompt_length: 512

benchmarks:
  - name: math500
    dataset: HuggingFaceH4/MATH-500
    split: test
    problem_key: problem
    answer_key: answer
    num_test_samples: 100
  - name: amc23
    dataset: math-ai/amc23
    split: test
    problem_key: question
    answer_key: answer
    num_test_samples: 40
  - name: aime24
    dataset: Maxwell-Jia/AIME_2024
    split: train
    problem_key: Problem
    answer_key: Answer
    num_test_samples: 30

generation:
  num_generations: 8
  temperature: 1.0
  max_new_tokens: 3072
  top_p: 1.0

training:
  learning_rate: 3.0e-6
  max_grad_norm: 1.0
  kl_coef: 0
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  num_train_epochs: 1
  max_steps: 300
  warmup_steps: 20
  logging_steps: 1
  save_steps: 50
  save_total_limit: 3
  eval_steps: 50
  skip_baseline_eval: false
  output_dir: ./outputs/sweep-qwen15b-${label}
  bf16: true
  seed: 42

reward:
  max_verifier_length: 3072
  token_level: false
  answer_only_cme: false
  no_box_penalty: 5

eval:
  max_new_tokens: 3072
  temperature: 0.0
  batch_size: 2

wandb:
  project: cme-grpo
  run_name: sweep-qwen15b-${label}-$(echo "$verifier" | tr '/' '-')
EOF

    log="logs/sweep_qwen15b_${label}.log"
    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] sweep qwen15b ${label}: ${verifier}"
    echo "  config: $cfg"
    echo "  log:    $log"
    echo "============================================================"

    python train.py --config "$cfg" 2>&1 | tee "$log"
done

echo ""
echo "============================================================"
echo "All Qwen2.5-1.5B base verifier-sweep runs complete."
echo "============================================================"
ls -1 outputs/sweep-qwen15b-* 2>/dev/null
