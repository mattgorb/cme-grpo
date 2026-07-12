#!/usr/bin/env bash
# Dense vs Scalar vs Reverse-KL — demonstrates the paper's DENSE token-level
# cross-tokenizer reward against (a) sequence-level scalar CME and (b) on-policy
# reverse-KL PG distillation. Two backbones: Qwen2.5-0.5B (base) and
# Llama-3.2-1B-Instruct. Verifier = gemma-4-E4B-it (different tokenizer -> the
# character-level alignment is exercised). Resumable; only win-rate judging needs API.
set -uo pipefail
JUDGES=${JUDGES:-gpt-5.2,claude-sonnet-4-6}
mkdir -p results logs

trained () { [ -f "$1/config.json" ] || [ -f "$1/.done" ]; }
train () {  # cfg log outdir
  if trained "$3"; then echo ">> SKIP train (done): $3"; return 0; fi
  rm -rf "$3"
  if python train_quality.py --config "$1" 2>&1 | tee "$2"; then touch "$3/.done";
  else echo "!! TRAIN FAILED: $1 — fix and re-run."; exit 1; fi
}
gen () {    # model outfile label
  if [ -s "$2" ]; then echo ">> SKIP gen (exists): $2"; return 0; fi
  python scripts/gen_outputs.py --model "$1" --out "$2" --label "$3" --num 200 --max-new-tokens 1024 \
    || { echo "!! GEN FAILED: $3"; exit 1; }
}
wr () {     # name outA outB labelA labelB
  if [ -f "results/$1/summary.json" ]; then echo ">> SKIP wr (done): $1"; return 0; fi
  python eval_head2head.py --outputs-a "$2" --label-a "$4" --outputs-b "$3" --label-b "$5" \
    --judges "$JUDGES" --out-dir "results/$1" || echo "!! wr $1 failed — continuing."
}

run_backbone () {   # tag reference_model reference_label
  local tag="$1" ref="$2" reflab="$3" R="results/dsr_${1}"
  mkdir -p "$R"
  train configs/config_dsr_${tag}_scalar.yaml    logs/dsr_${tag}_scalar.log    outputs/dsr-${tag}-scalar
  train configs/config_dsr_${tag}_dense.yaml     logs/dsr_${tag}_dense.log     outputs/dsr-${tag}-dense
  train configs/config_dsr_${tag}_reversekl.yaml logs/dsr_${tag}_reversekl.log outputs/dsr-${tag}-reversekl

  gen outputs/dsr-${tag}-dense     "$R/dense_outputs.json"     dense
  gen outputs/dsr-${tag}-scalar    "$R/scalar_outputs.json"    scalar
  gen outputs/dsr-${tag}-reversekl "$R/reversekl_outputs.json" reversekl
  gen "$ref"                       "$R/ref_outputs.json"       "$reflab"

  # Headlines: dense vs scalar (does the dense reward help?), dense vs reverse-KL
  # (CME vs distillation). Plus each vs the reference for context.
  wr dsr_${tag}_dense_vs_scalar "$R/dense_outputs.json"     "$R/scalar_outputs.json"    dense scalar
  wr dsr_${tag}_dense_vs_revkl  "$R/dense_outputs.json"     "$R/reversekl_outputs.json" dense reversekl
  wr dsr_${tag}_dense_vs_ref    "$R/dense_outputs.json"     "$R/ref_outputs.json"       dense "$reflab"
  wr dsr_${tag}_scalar_vs_ref   "$R/scalar_outputs.json"    "$R/ref_outputs.json"       scalar "$reflab"
  wr dsr_${tag}_revkl_vs_ref    "$R/reversekl_outputs.json" "$R/ref_outputs.json"       reversekl "$reflab"
}

run_backbone qwenbase Qwen/Qwen2.5-0.5B                base
run_backbone llamait  meta-llama/Llama-3.2-1B-Instruct instruct

echo "=================================================================="
echo " HEADLINES:"
for t in qwenbase llamait; do
  echo " [$t] dense vs scalar:"; cat results/dsr_${t}_dense_vs_scalar/summary.json 2>/dev/null
  echo " [$t] dense vs reverse-KL:"; cat results/dsr_${t}_dense_vs_revkl/summary.json 2>/dev/null
done
