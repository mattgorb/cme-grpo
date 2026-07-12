#!/usr/bin/env bash
# DECISIVE drift experiment (API-free): does CME collapse on a pretrained base,
# and does group_std mitigate it vs group_mean?
#   * 2 matched training runs (Qwen2.5-0.5B base, 400 steps, differ ONLY in
#     advantage_norm),
#   * greedy generations for both + base + teacher,
#   * drift metrics (length / distinct-n / self-BLEU / teacher-similarity).
# No judge calls anywhere -> costs zero API. Resumable: re-run to continue;
# completed training/outputs are skipped.
set -uo pipefail
mkdir -p results/drift logs

trained () { [ -f "$1/config.json" ] || [ -f "$1/.done" ]; }

train () {   # $1=config $2=log $3=outdir
  if trained "$3"; then echo ">> SKIP train (done): $3"; return 0; fi
  rm -rf "$3"   # clear partial checkpoints so train_quality doesn't auto-resume a corrupt one
  echo ">> TRAIN: $1 -> $3"
  if python train_quality.py --config "$1" 2>&1 | tee "$2"; then touch "$3/.done";
  else echo "!! TRAIN FAILED: $1 — fix and re-run to resume."; exit 1; fi
}

gen () {     # $1=model $2=outfile $3=label
  if [ -s "$2" ]; then echo ">> SKIP gen (exists): $2"; return 0; fi
  echo ">> GEN: $3 -> $2"
  python scripts/gen_outputs.py --model "$1" --out "$2" --label "$3" --num 200 --max-new-tokens 1024 \
    || { echo "!! GEN FAILED: $3"; exit 1; }
}

# --- run group_mean FIRST (the one predicted to collapse) ---
train configs/config_drift_qwenbase_groupmean.yaml logs/drift_groupmean.log outputs/drift-qwenbase-groupmean
train configs/config_drift_qwenbase_groupstd.yaml  logs/drift_groupstd.log  outputs/drift-qwenbase-groupstd

gen outputs/drift-qwenbase-groupmean results/drift/groupmean_outputs.json groupmean
gen outputs/drift-qwenbase-groupstd  results/drift/groupstd_outputs.json  groupstd
gen Qwen/Qwen2.5-0.5B                results/drift/base_outputs.json       base
gen google/gemma-4-E4B-it           results/drift/teacher_outputs.json    teacher

echo ">> DRIFT metrics"
python scripts/drift_metrics.py \
  --cond group_std=results/drift/groupstd_outputs.json \
  --cond group_mean=results/drift/groupmean_outputs.json \
  --cond base=results/drift/base_outputs.json \
  --teacher-outputs results/drift/teacher_outputs.json \
  --tokenizer Qwen/Qwen2.5-0.5B --out results/drift/decisive.json 2>&1 | tee logs/drift_decisive.log

# --- WIN RATES (needs API keys). Reuses the already-generated outputs -> judge-only,
# no regeneration. Confirms the diverse model (group_std) is also the BETTER one. ---
JUDGES=${JUDGES:-gpt-5.2,claude-sonnet-4-6}
wr () {  # $1=name $2=outfile_a $3=outfile_b $4=label_a $5=label_b
  if [ -f "results/$1/summary.json" ]; then echo ">> SKIP wr (done): $1"; return 0; fi
  python eval_head2head.py --outputs-a "$2" --label-a "$4" --outputs-b "$3" --label-b "$5" \
    --judges "$JUDGES" --out-dir "results/$1" || echo "!! wr $1 failed — continuing."
}
if [ -n "${OPENAI_API_KEY:-}" ] || [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  echo ">> WIN RATES (judge)"
  wr drift_std_vs_base  results/drift/groupstd_outputs.json  results/drift/base_outputs.json      group_std group_mean_base
  wr drift_mean_vs_base results/drift/groupmean_outputs.json results/drift/base_outputs.json      group_mean base
  wr drift_std_vs_mean  results/drift/groupstd_outputs.json  results/drift/groupmean_outputs.json group_std group_mean
else
  echo ">> SKIP win rates: no OPENAI_API_KEY/ANTHROPIC_API_KEY set (drift is done; export keys and re-run to add win rates)"
fi

echo "=================================================================="
echo " DECISIVE NUMBERS:"; cat results/drift/decisive.json
echo "=================================================================="
echo " Read: if group_mean is shorter / lower distinct-n / higher teacher_cos_sim"
echo " than group_std, that's the collapse group-norm mitigates -> story holds."
