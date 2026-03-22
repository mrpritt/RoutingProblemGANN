#!/usr/bin/env bash
# Run Lei et al. CVRP classical baseline for given node counts (including depot).
# Usage: ./scripts/run_baseline.sh [n1 n2 ...]
# Default sizes: 21 51 101  (= 20 50 100 customers + depot)
#
# Logs and checkpoints go to results/<run-id>/
# Matches Kool reduced batch settings: data_size=12800, batch_size=128, val_size=1000

set -euo pipefail

SIZES=${@:-21 51 101}
N_EPOCHS=${N_EPOCHS:-100}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

for N in $SIZES; do
    RUN_ID="vrp${N}_classical_$(date +%Y%m%dT%H%M%S)"
    LOG="results/${RUN_ID}/run.log"
    mkdir -p "results/${RUN_ID}"

    echo "=== Starting n=${N}, run=${RUN_ID}, n_epochs=${N_EPOCHS} ===" | tee "$LOG"

    QGAT_DECODER_BACKEND=classical \
    python -m VRP.VRP_Rollout_train \
        --n_nodes "$N" \
        --data_size 12800 \
        --batch_size 128 \
        --val_size 1000 \
        --n_epochs "$N_EPOCHS" \
        --output_dir results \
        --run_name "$RUN_ID" \
        2>&1 | tee -a "$LOG"

    echo "=== Finished n=${N} ===" | tee -a "$LOG"
done
