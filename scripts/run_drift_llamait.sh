#!/usr/bin/env bash
# DRIFT experiment #2 — Llama-3.2-1B-Instruct, 400 steps, matched group_std vs
# group_mean. Same design as run_drift_decisive.sh (Qwen base), but this backbone
# STARTS from an instruct checkpoint, so win rates are judged against the instruct
# starting point (did CME improve or degrade it) plus head-to-head. Teacher
# generations are reused from the Qwen drift run if already present.
# Resumable + API-free drift; win rates run only if API keys are set.
set -uo pipefail
mkdir -p results/drift_llamait logs

trained () { [ -f "$1/config.json" ] || [ -f "$1/.done" ]; }
train () {
  if trained "$3"; then echo ">> SKIP train (done): $3"; return 0; fi
  rm -rf "$3"
  echo ">> TRAIN: $1 -> $3"
  if python train_quality.py --config "$1" 2>&1 | tee "$2"; then touch "$3/.done";
  else echo "!! TRAIN FAILED: $1 — fix and re-run."; exit 1; fi
}
gen () {
  if [ -s "$2" ]; then echo ">> SKIP gen (exists): $2"; return 0; fi
  echo ">> GEN: $3 -> $2"
  python scripts/gen_outputs.py --model "$1" --out "$2" --label "$3" --num 200 --max-new-tokens 1024 \
    || { echo "!! GEN FAILED: $3"; exit 1; }
}

train configs/config_drift_llamait_groupmean.yaml logs/drift_llamait_groupmean.log outputs/drift-llamait-groupmean
train configs/config_drift_llamait_groupstd.yaml  logs/drift_llamait_groupstd.log  outputs/drift-llamait-groupstd

gen outputs/drift-llamait-groupmean       results/drift_llamait/groupmean_outputs.json groupmean
gen outputs/drift-llamait-groupstd        results/drift_llamait/groupstd_outputs.json  groupstd
gen meta-llama/Llama-3.2-1B-Instruct      results/drift_llamait/instruct_outputs.json  instruct
gen meta-llama/Llama-3.2-1B               results/drift_llamait/base_outputs.json      base
# reuse the teacher generations from the Qwen drift run (same 200 AlpacaEval prompts)
if [ -s results/drift/teacher_outputs.json ]; then
  cp -n results/drift/teacher_outputs.json results/drift_llamait/teacher_outputs.json
else
  gen google/gemma-4-E4B-it results/drift_llamait/teacher_outputs.json teacher
fi

echo ">> DRIFT metrics"
python scripts/drift_metrics.py \
  --cond group_std=results/drift_llamait/groupstd_outputs.json \
  --cond group_mean=results/drift_llamait/groupmean_outputs.json \
  --cond instruct=results/drift_llamait/instruct_outputs.json \
  --cond base=results/drift_llamait/base_outputs.json \
  --teacher-outputs results/drift_llamait/teacher_outputs.json \
  --tokenizer meta-llama/Llama-3.2-1B --out results/drift_llamait/decisive.json 2>&1 | tee logs/drift_llamait_metrics.log

# --- WIN RATES (judge-only, reuse outputs). Reference = the instruct start point. ---
JUDGES=${JUDGES:-gpt-5.2,claude-sonnet-4-6}
wr () {
  if [ -f "results/$1/summary.json" ]; then echo ">> SKIP wr (done): $1"; return 0; fi
  python eval_head2head.py --outputs-a "$2" --label-a "$4" --outputs-b "$3" --label-b "$5" \
    --judges "$JUDGES" --out-dir "results/$1" || echo "!! wr $1 failed — continuing."
}
if [ -n "${OPENAI_API_KEY:-}" ] || [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  echo ">> WIN RATES (judge)"
  wr drift_llamait_std_vs_instruct  results/drift_llamait/groupstd_outputs.json  results/drift_llamait/instruct_outputs.json group_std  instruct
  wr drift_llamait_mean_vs_instruct results/drift_llamait/groupmean_outputs.json results/drift_llamait/instruct_outputs.json group_mean instruct
  wr drift_llamait_std_vs_mean      results/drift_llamait/groupstd_outputs.json  results/drift_llamait/groupmean_outputs.json group_std  group_mean
else
  echo ">> SKIP win rates: no OPENAI_API_KEY/ANTHROPIC_API_KEY set (export keys and re-run to add them)"
fi

echo "=================================================================="
echo " LLAMA-IT DRIFT:"; cat results/drift_llamait/decisive.json
echo "=================================================================="
