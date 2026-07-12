#!/usr/bin/env bash
# Resumable rebuttal runner. Safe to re-run after any crash/quota/disconnect:
#   * teacher-gen  : skipped if the JSONL already exists
#   * a training run: skipped if its output dir already has a saved model
#   * an eval      : skipped if results/<name>/summary.json already exists
# Training failure => abort (fix, then re-run to resume). Eval failure => warn &
# continue (re-run later to retry) so a bad eval never discards trained models.
#
# Set these to where condition-A's cached generations live, then run in tmux:
A_QWEN=${A_QWEN:-/path/to/cme-grpo-quality-config6/quality_eval}
A_LLAMA=${A_LLAMA:-/path/to/cme-grpo-quality-config8/quality_eval}
JUDGES=${JUDGES:-gpt-5.2,claude-sonnet-4-6}

set -uo pipefail   # NOT -e: per-step handling below decides what aborts
mkdir -p results logs data

# ---- helpers ---------------------------------------------------------------
trained () { [ -f "$1/config.json" ] || [ -f "$1/.done" ]; }   # $1 = output dir

train () {   # $1=config  $2=logfile  $3=output_dir
  if trained "$3"; then echo ">> SKIP train (already done): $3"; return 0; fi
  # Not done -> start FRESH. Clear any partial/corrupt checkpoints, else
  # train_quality.py auto-resumes from an incomplete checkpoint and crashes.
  rm -rf "$3"
  echo ">> TRAIN: $1  ->  $3"
  if python train_quality.py --config "$1" 2>&1 | tee "$2"; then
    touch "$3/.done"
  else
    echo "!! TRAIN FAILED: $1  — fix the cause, then re-run this script to resume."
    exit 1
  fi
}

evaled () { [ -f "results/$1/summary.json" ]; }   # $1 = result name

h2h () {     # $1=name  rest=eval_head2head args (out-dir/judges added here)
  local name="$1"; shift
  if evaled "$name"; then echo ">> SKIP eval (already done): results/$name"; return 0; fi
  echo ">> EVAL: $name"
  python eval_head2head.py "$@" --judges "$JUDGES" --out-dir "results/$name" \
    || echo "!! EVAL FAILED: $name — continuing; re-run script to retry."
}

# ---- Exp 3 teacher generation (skip if present) ----------------------------
TEACHER_OUT=data/teacher_sft_qwenprompts.jsonl
if [ -s "$TEACHER_OUT" ]; then
  echo ">> SKIP teacher-gen (found $(wc -l < "$TEACHER_OUT") rows)"
else
  echo ">> TEACHER GEN"
  python scripts/gen_teacher_data.py \
    --config configs/config_exp1c_qwen_revkl.yaml \
    --teacher google/gemma-4-E4B-it --out "$TEACHER_OUT" \
    --temperature 0.7 --max-new-tokens 1024 2>&1 | tee logs/teacher_gen.log
fi

# ---- Exp 1C: reverse-KL (group_mean) + interleaved evals -------------------
train configs/config_exp1c_qwen_revkl.yaml  logs/exp1c_qwen.log  outputs/exp1c-qwen-revkl
h2h exp1_qwen_AvsC   --outputs-a "$A_QWEN/finetuned_outputs.json" --label-a CME_group_std \
                     --model-b outputs/exp1c-qwen-revkl           --label-b revKL_group_mean
h2h exp1_qwen_Cvsbase --model-a outputs/exp1c-qwen-revkl --label-a revKL \
                      --outputs-b "$A_QWEN/base_outputs.json"     --label-b base

train configs/config_exp1c_llama_revkl.yaml logs/exp1c_llama.log outputs/exp1c-llama-revkl
if [ -s "$A_LLAMA/finetuned_outputs.json" ]; then
  h2h exp1_llama_AvsC --outputs-a "$A_LLAMA/finetuned_outputs.json" --label-a CME_group_std \
                      --model-b outputs/exp1c-llama-revkl           --label-b revKL_group_mean
else
  echo ">> SKIP exp1_llama_AvsC: A_LLAMA/finetuned_outputs.json missing/empty (Llama A not recoverable)"
fi
# base generated fresh (cached llama base may be empty), so this is robust:
h2h exp1_llama_Cvsbase --model-a outputs/exp1c-llama-revkl --label-a revKL \
                       --model-b meta-llama/Llama-3.2-1B  --label-b base

# ---- Exp 2: beta sweep (each vs A=beta0.01) --------------------------------
train configs/config_exp2_qwen_beta0.yaml   logs/exp2_beta0.log  outputs/exp2-qwen-beta0
h2h exp2_beta0_vs_A  --outputs-a "$A_QWEN/finetuned_outputs.json" --label-a beta0.01 \
                     --model-b outputs/exp2-qwen-beta0            --label-b beta0

train configs/config_exp2_qwen_beta0.1.yaml logs/exp2_beta01.log outputs/exp2-qwen-beta0.1
h2h exp2_beta01_vs_A --outputs-a "$A_QWEN/finetuned_outputs.json" --label-a beta0.01 \
                     --model-b outputs/exp2-qwen-beta0.1          --label-b beta0.1

# ---- Exp 3: forward-KD SFT + evals -----------------------------------------
if trained outputs/exp3-sft-kd-qwen; then
  echo ">> SKIP train (already done): outputs/exp3-sft-kd-qwen"
else
  rm -rf outputs/exp3-sft-kd-qwen
  echo ">> TRAIN SFT -> outputs/exp3-sft-kd-qwen"
  if python train_sft.py --data "$TEACHER_OUT" --student Qwen/Qwen2.5-0.5B \
       --output-dir outputs/exp3-sft-kd-qwen --epochs 1 --lr 1e-5 --max-steps 600 \
       2>&1 | tee logs/exp3_sft.log; then
    touch outputs/exp3-sft-kd-qwen/.done
  else
    echo "!! SFT FAILED — fix and re-run to resume."; exit 1
  fi
fi
h2h exp3_kd_vs_base --model-a outputs/exp3-sft-kd-qwen --label-a SFT_forwardKD \
                    --outputs-b "$A_QWEN/base_outputs.json" --label-b base
h2h exp3_kd_vs_A    --model-a outputs/exp3-sft-kd-qwen --label-a SFT_forwardKD \
                    --outputs-b "$A_QWEN/finetuned_outputs.json" --label-b CME_group_std

# ---- Exp 4: teacher-on-AlpacaEval gen (for drift) + drift metrics ----------
h2h teacher_gen --model-a google/gemma-4-E4B-it --label-a teacher \
                --outputs-b "$A_QWEN/finetuned_outputs.json" --label-b CME_group_std

if [ -f results/drift/qwen_drift.json ]; then
  echo ">> SKIP drift (results/drift/qwen_drift.json exists)"
else
  echo ">> DRIFT metrics"
  python scripts/drift_metrics.py \
    --cond A_group_std="$A_QWEN/finetuned_outputs.json" \
    --cond C_group_mean=results/exp1_qwen_AvsC/revKL_group_mean_outputs.json \
    --cond beta0=results/exp2_beta0_vs_A/beta0_outputs.json \
    --cond beta0.1=results/exp2_beta01_vs_A/beta0.1_outputs.json \
    --cond SFT_KD=results/exp3_kd_vs_base/SFT_forwardKD_outputs.json \
    --teacher-outputs results/teacher_gen/teacher_outputs.json \
    --tokenizer Qwen/Qwen2.5-0.5B \
    --out results/drift/qwen_drift.json 2>&1 | tee logs/drift.log \
    || echo "!! drift failed — continuing."
fi

echo "=== DONE. Headline: results/exp1_qwen_AvsC/summary.json ==="
echo "=== Re-run this script any time; completed steps are skipped. ==="
