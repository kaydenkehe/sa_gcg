#!/usr/bin/env bash
# EVAL_PLAN §8 SA-GCG pilot: 12 cells × 5 behaviors × 1 seed × 15 min budget.
# Skips cells whose suffix.json already exists, so safe to re-run after a crash.
#
# Usage:
#   bash scripts/run_pilot.sh [direction.pt] [model_id] [pilot_root]
#
# Env-var overrides (take precedence over positional args):
#   PILOT_ROOT   target directory  (default: runs/pilot/sa_gcg)
#   BUDGET_S     wall-clock per cell, seconds  (default: 900 = 15 min)
#   LIMIT        behaviors per cell  (default: 5)
#
# Example — long 1-hr/cell pilot into its own tree:
#   PILOT_ROOT=runs/pilot_long/sa_gcg BUDGET_S=3600 bash scripts/run_pilot.sh
#
# Example — n=25 universal pilot (paper sizing) at 1 hr/cell:
#   PILOT_ROOT=runs/pilot_n25/sa_gcg BUDGET_S=3600 LIMIT=25 bash scripts/run_pilot.sh

set -uo pipefail

DIR="${1:-runs/full/direction/llama2-7b-chat.pt}"
MODEL="${2:-meta-llama/Llama-2-7b-chat-hf}"
OUT_ROOT="${PILOT_ROOT:-${3:-runs/pilot/sa_gcg}}"

if [[ ! -f "$DIR" ]]; then
    echo "ERROR: direction file not found: $DIR" >&2
    exit 2
fi

SCOPES=(single layer global)
LAMBDAS=(0.5 1.0)
POLISHES=(0 50)

BUDGET_S="${BUDGET_S:-900}"   # 15 min default; override via env var
N_STEPS_CEIL=100000     # ceiling; budget terminates first
LIMIT="${LIMIT:-5}"     # behaviors per cell; override via env var
SEED=0

n_total=0
n_skipped=0
n_failed=0
failed_cells=()
t_start=$(date +%s)

for scope in "${SCOPES[@]}"; do
    for lam in "${LAMBDAS[@]}"; do
        for polish in "${POLISHES[@]}"; do
            n_total=$((n_total + 1))
            cell="scope_${scope}_lam_${lam}_polish_${polish}"
            out="$OUT_ROOT/$cell/llama2/seed_$SEED"

            if [[ -f "$out/suffix.json" ]]; then
                echo "[$n_total/12] SKIP $cell (suffix.json exists)"
                n_skipped=$((n_skipped + 1))
                continue
            fi

            mkdir -p "$out"
            t0=$(date +%s)
            echo "[$n_total/12] RUN  $cell  ($(date +%H:%M:%S))"

            sa-gcg-attack \
                --model "$MODEL" \
                --attack sa_gcg --universal \
                --dataset advbench --limit "$LIMIT" \
                --direction-path "$DIR" \
                --activation-scope "$scope" \
                --composite-lambda "$lam" \
                --polish-steps "$polish" \
                --n-steps "$N_STEPS_CEIL" \
                --wall-clock-budget-s "$BUDGET_S" \
                --batch-size 256 --suffix-length 20 \
                --out-dir "$out" --seed "$SEED" \
                > "$out/attack.log" 2>&1

            rc=$?
            t1=$(date +%s)
            dt=$((t1 - t0))
            if [[ $rc -ne 0 ]]; then
                echo "       FAIL (rc=$rc, ${dt}s, see $out/attack.log)"
                n_failed=$((n_failed + 1))
                failed_cells+=("$cell")
            else
                echo "       OK   (${dt}s)"
            fi
        done
    done
done

t_end=$(date +%s)
total_min=$(( (t_end - t_start) / 60 ))

echo
echo "================================================================"
echo "PILOT DONE"
echo "  total cells    : $n_total"
echo "  ran            : $((n_total - n_skipped - n_failed))"
echo "  skipped (cached): $n_skipped"
echo "  failed         : $n_failed"
echo "  wall-clock     : ${total_min} min"
if [[ $n_failed -gt 0 ]]; then
    echo "  failed cells   : ${failed_cells[*]}"
fi
echo "================================================================"

[[ $n_failed -eq 0 ]] && exit 0 || exit 1
