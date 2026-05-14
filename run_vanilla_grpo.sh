#!/usr/bin/env bash
# Train vanilla GRPO (binary correctness reward) on three configs sequentially.
# Output dirs and wandb run names are auto-rewritten cme-grpo-* → vanilla-grpo-*
# inside train_vanilla_grpo.py.

set -euo pipefail

mkdir -p logs

CONFIGS=(
    "good_configs/config2.yaml"
    "config3.yaml"
    "config3b.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    if [[ ! -f "$cfg" ]]; then
        echo "ERROR: config not found: $cfg" >&2
        exit 1
    fi

    name=$(basename "$cfg" .yaml)
    log="logs/vanilla_${name}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] vanilla GRPO: $cfg"
    echo "  log: $log"
    echo "============================================================"

    python train_vanilla_grpo.py --config "$cfg" 2>&1 | tee "$log"
done

echo ""
echo "============================================================"
echo "All vanilla GRPO runs complete."
echo "============================================================"
ls -1 outputs/vanilla-grpo-* 2>/dev/null
