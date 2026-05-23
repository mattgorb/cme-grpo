#!/usr/bin/env bash
# Run the two RENT instruct-model experiments sequentially.
# Both train for 1000 actual steps with the 5000-step LR schedule (mean-pooled
# entropy reward, kl_coef=0.05).
#   rent2: Qwen2.5-0.5B-Instruct
#   rent3: Llama-3.2-1B-Instruct

set -euo pipefail

mkdir -p logs

CONFIGS=(
    "configs/config_quality_rent2.yaml"
    "configs/config_quality_rent3.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    if [[ ! -f "$cfg" ]]; then
        echo "ERROR: config not found: $cfg" >&2
        exit 1
    fi

    name=$(basename "$cfg" .yaml)
    log="logs/${name}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] RENT run: $cfg"
    echo "  log: $log"
    echo "============================================================"

    python train_quality.py --config "$cfg" 2>&1 | tee "$log"
done

echo ""
echo "============================================================"
echo "All RENT runs complete."
echo "============================================================"
ls -1 outputs/rent2-* outputs/rent3-* 2>/dev/null
