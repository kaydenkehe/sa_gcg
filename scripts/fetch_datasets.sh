#!/usr/bin/env bash
# Fetch the four held-out evaluation datasets used by EVAL_PLAN.md.
# Usage:  bash scripts/fetch_datasets.sh [target_dir]
# Default target: src/sa_gcg/data/raw
set -euo pipefail

TARGET="${1:-$(dirname "$0")/../src/sa_gcg/data/raw}"
mkdir -p "$TARGET"
cd "$TARGET"

echo "Fetching into $(pwd) ..."

# AdvBench (Zou et al. 2023)
if [[ ! -f advbench.csv ]]; then
    curl -fsSL -o advbench.csv \
        https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv
fi

# HarmBench test split (Mazeika et al. 2024)
if [[ ! -f harmbench_test.csv ]]; then
    curl -fsSL -o harmbench_test.csv \
        https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_test.csv
fi

# JailbreakBench (Chao et al. 2024) — via Hugging Face datasets repo
if [[ ! -f jailbreakbench.csv ]]; then
    curl -fsSL -o jailbreakbench.csv \
        https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/behaviors.csv \
        || echo "WARN: HF download failed; install 'huggingface_hub' and run 'huggingface-cli download JailbreakBench/JBB-Behaviors --repo-type dataset'"
fi

# StrongREJECT (Souly et al. 2024)
if [[ ! -f strongreject.csv ]]; then
    curl -fsSL -o strongreject.csv \
        https://raw.githubusercontent.com/dsbowen/strong_reject/main/strong_reject/eval_files/strongreject_dataset.csv
fi

echo "Done. Files in $TARGET:"
ls -la
