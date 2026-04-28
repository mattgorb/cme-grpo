#!/usr/bin/env bash
# Verifier capability sweep.
# Generator fixed at Qwen2.5-Math-1.5B, MATH-lighteval, 600 steps, eval every 200.
# Hyperparameters: G=8, beta=0, epsilon=0.2 (TRL default), lr=3e-6.
# Reports pass@1 on MATH-500, AMC-23, AIME-24 via PeriodicEvalCallback.

set -euo pipefail

mkdir -p logs configs/sweep

# label : MATH-500 pass@1 : verifier model
VERIFIERS=(
    "v1:83:Qwen/Qwen2.5-Math-7B-Instruct"
    "v2:75:Qwen/Qwen2.5-Math-1.5B-Instruct"
    "v3:65:Qwen/Qwen2.5-Math-1.5B"
    "v4:35:Qwen/Qwen2.5-0.5B-Instruct"
    "v5:20:Qwen/Qwen2.5-0.5B"
)

for entry in "${VERIFIERS[@]}"; do
    label="${entry%%:*}"
    rest="${entry#*:}"
    pass="${rest%%:*}"
    verifier="${rest#*:}"
    cfg="configs/sweep/${label}.yaml"

    cat > "$cfg" <<EOF
# Verifier sweep ${label} — ${verifier} (~${pass}% MATH-500 pass@1)
model:
  generator: Qwen/Qwen2.5-Math-1.5B
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
  max_steps: 600
  warmup_steps: 20
  logging_steps: 1
  save_steps: 200
  save_total_limit: 3
  eval_steps: 200
  skip_baseline_eval: false
  output_dir: ./outputs/sweep-${label}
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
  run_name: sweep-${label}-$(echo "$verifier" | tr '/' '-')
EOF

    log="logs/sweep_${label}.log"
    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] sweep ${label}: ${verifier}"
    echo "  config: $cfg"
    echo "  log:    $log"
    echo "============================================================"

    python train.py --config "$cfg" 2>&1 | tee "$log"
done

echo ""
echo "============================================================"
echo "All verifier sweep runs complete."
echo "============================================================"
ls -1 outputs/sweep-* 2>/dev/null
