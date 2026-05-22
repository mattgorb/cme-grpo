#!/usr/bin/env bash
# Verifier capability sweep using the config4 (Llama generator) setup.
# Three verifier conditions: random, small Qwen-Math, big Qwen-Math.
# Trains for 600 actual steps but keeps the config4 LR schedule (max_steps=5000)
# via the early_stop_steps callback in train.py.

set -euo pipefail

mkdir -p logs configs/sweep_llama

# label : MATH-500 pass@1 (rough) : verifier model
VERIFIERS=(
    "l1:random:hf-internal-testing/tiny-random-LlamaForCausalLM"
    "l2:65:Qwen/Qwen2.5-Math-1.5B"
    "l3:83:Qwen/Qwen2.5-Math-7B-Instruct"
)

for entry in "${VERIFIERS[@]}"; do
    label="${entry%%:*}"
    rest="${entry#*:}"
    pass="${rest%%:*}"
    verifier="${rest#*:}"
    cfg="configs/sweep_llama/${label}.yaml"

    cat > "$cfg" <<EOF
# Llama verifier sweep ${label} — ${verifier} (~${pass}% MATH-500 pass@1)
# Mirrors config4 hyperparameters; trains 600 steps with the 5000-step LR schedule.
model:
  generator: meta-llama/Llama-3.2-1B-Instruct
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
  max_steps: 5000          # scheduler horizon (matches config4)
  early_stop_steps: 600    # actually stop here
  warmup_steps: 20
  logging_steps: 1
  save_steps: 25
  save_total_limit: 3
  eval_steps: 25
  skip_baseline_eval: true
  output_dir: ./outputs/sweep-llama-${label}
  bf16: true
  seed: 42

reward:
  max_verifier_length: 3072
  token_level: false
  answer_only_cme: false
  no_box_penalty: 5
  reward_metric: 'predictive_entropy'

eval:
  max_new_tokens: 3072
  temperature: 0.0
  batch_size: 2

wandb:
  project: cme-grpo
  run_name: sweep-llama-${label}-$(echo "$verifier" | tr '/' '-')
EOF

    log="logs/sweep_llama_${label}.log"
    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] sweep-llama ${label}: ${verifier}"
    echo "  config: $cfg"
    echo "  log:    $log"
    echo "============================================================"

    python train.py --config "$cfg" 2>&1 | tee "$log"
done

echo ""
echo "============================================================"
echo "All Llama verifier sweep runs complete."
echo "============================================================"
ls -1 outputs/sweep-llama-* 2>/dev/null
