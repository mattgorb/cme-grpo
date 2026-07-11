#!/usr/bin/env bash
# Overnight rebuttal runner (corrected plan). Run on the GPU box (L40S, 48GB).
#
# Conditions reflect what the code ACTUALLY does:
#   * open-ended CME is SEQUENCE-level (train_quality.build_quality_reward_fn),
#   * condition A used beta=0.01 and verifier gemma-4-E4B-it (config_quality6/8),
#   * condition A weights aren't saved, but A's greedy generations ARE cached and
#     are reused for judging (no A re-run needed).
#
# Set these to where A's cached eval lives on THIS machine before running:
A_QWEN=${A_QWEN:-/path/to/cme-grpo-quality-config6/quality_eval}   # has finetuned_outputs.json, base_outputs.json, instruct_outputs.json
A_LLAMA=${A_LLAMA:-/path/to/cme-grpo-quality-config8/quality_eval}
JUDGES=${JUDGES:-gpt-5.2,claude-sonnet-4-6}
set -euo pipefail
mkdir -p results logs data

TEACHER_OUT=data/teacher_sft_qwenprompts.jsonl
if [ -s "$TEACHER_OUT" ]; then
  echo "=== Exp 3 teacher generation: SKIP (found $(wc -l < "$TEACHER_OUT") rows in $TEACHER_OUT) ==="
else
  echo "=== Exp 3 teacher generation (light, run first / overnight day 0) ==="
  python scripts/gen_teacher_data.py \
    --config configs/config_exp1c_qwen_revkl.yaml \
    --teacher google/gemma-4-E4B-it \
    --out "$TEACHER_OUT" \
    --temperature 0.7 --max-new-tokens 1024 2>&1 | tee logs/teacher_gen.log
fi

echo "=== Exp 1 C: reverse-KL PG (group_mean) training ==="
python train_quality.py --config configs/config_exp1c_qwen_revkl.yaml  2>&1 | tee logs/exp1c_qwen.log
python train_quality.py --config configs/config_exp1c_llama_revkl.yaml 2>&1 | tee logs/exp1c_llama.log

echo "=== Exp 2 beta sweep training (Qwen) ==="
python train_quality.py --config configs/config_exp2_qwen_beta0.yaml   2>&1 | tee logs/exp2_beta0.log
python train_quality.py --config configs/config_exp2_qwen_beta0.1.yaml 2>&1 | tee logs/exp2_beta01.log

echo "=== Exp 3 forward-KD SFT ==="
python train_sft.py --data data/teacher_sft_qwenprompts.jsonl \
  --student Qwen/Qwen2.5-0.5B --output-dir ./outputs/exp3-sft-kd-qwen \
  --epochs 1 --lr 1e-5 --max-steps 600 2>&1 | tee logs/exp3_sft.log

echo "=== Exp 1 evals (reuse cached A; generate C) ==="
# Core exhibit: A (CME group_std) vs C (reverse-KL group_mean)
python eval_head2head.py \
  --outputs-a "$A_QWEN/finetuned_outputs.json"     --label-a CME_group_std \
  --model-b ./outputs/exp1c-qwen-revkl             --label-b revKL_group_mean \
  --judges "$JUDGES" --out-dir results/exp1_qwen_AvsC 2>&1 | tee logs/eval_exp1_qwen_AvsC.log
# C vs base (reuse cached base outputs)
python eval_head2head.py \
  --model-a ./outputs/exp1c-qwen-revkl --label-a revKL \
  --outputs-b "$A_QWEN/base_outputs.json" --label-b base \
  --judges "$JUDGES" --out-dir results/exp1_qwen_Cvsbase 2>&1 | tee logs/eval_exp1_qwen_Cvsbase.log
# Llama A vs C
python eval_head2head.py \
  --outputs-a "$A_LLAMA/finetuned_outputs.json" --label-a CME_group_std \
  --model-b ./outputs/exp1c-llama-revkl         --label-b revKL_group_mean \
  --judges "$JUDGES" --out-dir results/exp1_llama_AvsC 2>&1 | tee logs/eval_exp1_llama_AvsC.log

echo "=== Exp 2 evals (each beta vs A=beta0.01 and vs base) ==="
python eval_head2head.py \
  --outputs-a "$A_QWEN/finetuned_outputs.json" --label-a beta0.01 \
  --model-b ./outputs/exp2-qwen-beta0 --label-b beta0 \
  --judges "$JUDGES" --out-dir results/exp2_beta0_vs_A 2>&1 | tee logs/eval_exp2_beta0.log
python eval_head2head.py \
  --outputs-a "$A_QWEN/finetuned_outputs.json" --label-a beta0.01 \
  --model-b ./outputs/exp2-qwen-beta0.1 --label-b beta0.1 \
  --judges "$JUDGES" --out-dir results/exp2_beta01_vs_A 2>&1 | tee logs/eval_exp2_beta01.log

echo "=== Exp 3 eval (SFT-KD vs base, vs A) ==="
python eval_head2head.py \
  --model-a ./outputs/exp3-sft-kd-qwen --label-a SFT_forwardKD \
  --outputs-b "$A_QWEN/base_outputs.json" --label-b base \
  --judges "$JUDGES" --out-dir results/exp3_kd_vs_base 2>&1 | tee logs/eval_exp3_kd_base.log
python eval_head2head.py \
  --model-a ./outputs/exp3-sft-kd-qwen --label-a SFT_forwardKD \
  --outputs-b "$A_QWEN/finetuned_outputs.json" --label-b CME_group_std \
  --judges "$JUDGES" --out-dir results/exp3_kd_vs_A 2>&1 | tee logs/eval_exp3_kd_A.log

echo "=== Exp 4 drift metrics (CPU) ==="
# Teacher's own generations on the SAME prompt set as A, for teacher-similarity.
python eval_head2head.py \
  --model-a google/gemma-4-E4B-it --label-a teacher \
  --outputs-b "$A_QWEN/finetuned_outputs.json" --label-b CME_group_std \
  --judges "$JUDGES" --out-dir results/teacher_gen 2>&1 | tee logs/teacher_on_alpaca.log || true
python scripts/drift_metrics.py \
  --cond A_group_std="$A_QWEN/finetuned_outputs.json" \
  --cond C_group_mean=results/exp1_qwen_AvsC/revKL_group_mean_outputs.json \
  --cond beta0=results/exp2_beta0_vs_A/beta0_outputs.json \
  --cond beta0.1=results/exp2_beta01_vs_A/beta0.1_outputs.json \
  --cond SFT_KD=results/exp3_kd_vs_base/SFT_forwardKD_outputs.json \
  --teacher-outputs results/teacher_gen/teacher_outputs.json \
  --tokenizer Qwen/Qwen2.5-0.5B \
  --out results/drift/qwen_drift.json 2>&1 | tee logs/drift.log

echo "=== DONE. Headline: results/exp1_qwen_AvsC/summary.json ==="
