#!/usr/bin/env bash
# Quality verifier-capability sweep — Qwen/Qwen2.5-0.5B (base) as generator.
# Three verifier conditions:
#   q1: random-weighted tiny model (no signal control)
#   q2: gemma-3-270m-it (very small but real-weighted)
#   q3: Qwen2.5-1.5B-Instruct (1.5B, real)
# Training: 600 steps, eval every 150, static (constant) LR.
# Other params inherited from config_quality6.

set -euo pipefail

mkdir -p logs configs/sweep_quality

# label : description : verifier model
VERIFIERS=(
    "q1:random:hf-internal-testing/tiny-random-LlamaForCausalLM"
    "q2:gemma-270m:google/gemma-3-270m-it"
    "q3:qwen-1.5b:Qwen/Qwen2.5-1.5B-Instruct"
)

for entry in "${VERIFIERS[@]}"; do
    label="${entry%%:*}"
    rest="${entry#*:}"
    desc="${rest%%:*}"
    verifier="${rest#*:}"
    cfg="configs/sweep_quality/${label}.yaml"

    cat > "$cfg" <<EOF
# Quality verifier sweep ${label} — ${desc} (${verifier})
model:
  generator: Qwen/Qwen2.5-0.5B
  verifier: ${verifier}
  # three-way LLM judge comparison
  base: Qwen/Qwen2.5-0.5B
  instruct: Qwen/Qwen2.5-0.5B-Instruct

data:
  train_dataset: openbmb/UltraFeedback
  max_train_samples: 5000
  max_prompt_length: 512

generation:
  num_generations: 8
  temperature: 1.0
  max_new_tokens: 2048
  top_p: 1.0

training:
  learning_rate: 3.0e-6
  lr_scheduler_type: constant
  max_grad_norm: 1.0
  kl_coef: 0.01
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  num_train_epochs: 1
  max_steps: 1000
  warmup_steps: 0
  logging_steps: 1
  save_steps: 200
  save_total_limit: 3
  eval_steps: 200
  skip_baseline_eval: false
  output_dir: ./outputs/sweep-quality-${label}
  bf16: true
  seed: 42

reward:
  max_verifier_length: 2048
  token_level: true
  answer_only_cme: false
  reward_metric: entropy

eval:
  max_new_tokens: 2048
  batch_size: 4
  judge_model: gpt-5.2
  judge_num_samples: 50

wandb:
  project: cme-grpo
  run_name: sweep-quality-${label}-${desc}
EOF

    log="logs/sweep_quality_${label}.log"
    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] quality sweep ${label}: ${verifier}"
    echo "  config: $cfg"
    echo "  log:    $log"
    echo "============================================================"

    python train_quality.py --config "$cfg" 2>&1 | tee "$log"
done

echo ""
echo "============================================================"
echo "All quality verifier-sweep runs complete."
echo "============================================================"
ls -1 outputs/sweep-quality-* 2>/dev/null
