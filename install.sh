#!/usr/bin/env bash
# Setup script for cme-grpo on a fresh RunPod GPU instance.
# Redirects all caches to /workspace (persistent volume), cleans old caches in
# /root, removes torchvision (compiled against wrong torch), and installs deps.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
CACHE_ROOT="$WORKSPACE/.cache"

echo "[install] redirecting caches to $CACHE_ROOT"
mkdir -p \
    "$CACHE_ROOT/huggingface" \
    "$CACHE_ROOT/huggingface/hub" \
    "$CACHE_ROOT/huggingface/datasets" \
    "$CACHE_ROOT/torch" \
    "$CACHE_ROOT/pip" \
    "$CACHE_ROOT/wandb"

# Write env vars to ~/.bashrc so they persist across new shells (tmux, etc).
ENV_BLOCK_MARKER="# === cme-grpo cache redirects ==="
if ! grep -q "$ENV_BLOCK_MARKER" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc <<EOF

$ENV_BLOCK_MARKER
export HF_HOME=$CACHE_ROOT/huggingface
export HUGGINGFACE_HUB_CACHE=$CACHE_ROOT/huggingface/hub
export TRANSFORMERS_CACHE=$CACHE_ROOT/huggingface/hub
export HF_DATASETS_CACHE=$CACHE_ROOT/huggingface/datasets
export TORCH_HOME=$CACHE_ROOT/torch
export PIP_CACHE_DIR=$CACHE_ROOT/pip
export XDG_CACHE_HOME=$CACHE_ROOT
export WANDB_DIR=$WORKSPACE/cme-grpo/wandb
export WANDB_CACHE_DIR=$CACHE_ROOT/wandb
# === end cme-grpo ===
EOF
    echo "[install] wrote env vars to ~/.bashrc"
fi

# Export for THIS shell session too (so the rest of this script uses them).
export HF_HOME="$CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/hub"
export HF_DATASETS_CACHE="$CACHE_ROOT/huggingface/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export XDG_CACHE_HOME="$CACHE_ROOT"
export WANDB_DIR="$WORKSPACE/cme-grpo/wandb"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"

echo "[install] cleaning old caches in /root (these are on the small root disk)"
rm -rf /root/.cache/huggingface || true
rm -rf /root/.cache/pip || true
rm -rf /root/.cache/torch || true
rm -rf /tmp/huggingface_* || true

echo "[install] removing torchvision (incompatible with bundled torch)"
pip uninstall -y torchvision torchaudio || true

echo "[install] installing requirements"
pip install -r "$(dirname "$0")/requirements.txt"

echo "[install] sanity-checking imports"
python -c "from transformers import PreTrainedModel, TrainerCallback; from peft import PeftModel; from trl import GRPOTrainer; print('[install] ok')"

echo
echo "[install] done. To use in this shell now: source ~/.bashrc"
echo "[install] disk usage:"
df -h "$WORKSPACE" / 2>/dev/null | head -3
