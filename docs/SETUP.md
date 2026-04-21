# Remote-runner setup guide

This walks you through everything needed to run `sa_gcg` end-to-end on a fresh
Linux box with one or more NVIDIA GPUs. Follow the sections in order. Each
later section depends only on what came before.

The ordered test list is at the bottom (Section 9). Read sections 1–8 first,
then work the test list top-down — every test only depends on prior tests
having passed.

---

## 1. Hardware & system prerequisites

Minimum viable: 1× GPU with ≥24 GB VRAM (A10, A6000, RTX 6000 Ada, A100-40G).
Used by:

- The 7B target model (Vicuna-7B / Llama-2-7B-Chat) — fp16, ~14 GB.
- HarmBench classifier (`cais/HarmBench-Llama-2-13b-cls`) — fp16, ~26 GB.

If you only have one 24 GB card, run attacks and HarmBench scoring in
**separate jobs** (the runner unloads in between). To run them concurrently,
use either: 2 GPUs (set `CUDA_VISIBLE_DEVICES` per job) or 1 GPU ≥48 GB.

For the full 13B/70B open-transfer eval, see Section 4 ("model cache size").

System packages (Debian/Ubuntu names; adapt for your distro):

```bash
sudo apt-get update
sudo apt-get install -y \
    git curl build-essential \
    python3.11 python3.11-venv python3.11-dev
nvidia-smi          # confirm GPU is visible
nvcc --version      # confirm CUDA toolkit (need >=11.8)
```

Python ≥3.10 is required. 3.11 is the recommended sweet spot, though 3.13
also works.

---

## 2. Clone repo and create the venv

```bash
# 1. Repo
git clone <YOUR-REMOTE>/sa_gcg.git
cd sa_gcg

# 2. Virtualenv
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel

# 3. Install (editable) with all extras you'll need
pip install -e '.[api,dev]'

# Optional but recommended (StrongREJECT judge metadata for some Souly fields):
pip install -e '.[strongreject]'
```

What the extras give you:

| Extra            | Adds                                                             |
|------------------|------------------------------------------------------------------|
| `api`            | `openai`, `anthropic`, `google-generativeai` (closed transfer)   |
| `dev`            | `pytest`, `pytest-mock`, `ruff`                                  |
| `strongreject`   | `strong_reject` from upstream git (only needed if you want their canonical helpers; the in-tree judge works without it) |

After `pip install`, sanity check:

```bash
python -c "import sa_gcg, torch; print('ok', torch.cuda.is_available())"
```

You should see `ok True`. If `False`, either the GPU isn't visible to the
container, or PyTorch installed a CPU wheel — re-install with the CUDA index:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 'torch>=2.1'
```

---

## 3. Authenticate to model + API providers

### 3.1 Hugging Face (required)

The Llama-2 / Llama-3 / Llama-Guard models and HarmBench classifier are
**gated** — you need a HF account that has accepted the licenses.

```bash
pip install huggingface_hub
huggingface-cli login                  # paste a read-token from huggingface.co/settings/tokens
# or, non-interactive:
export HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxx
```

Then visit each model page and click "Access repository" (one-time per model):

- `meta-llama/Llama-2-7b-chat-hf`
- `meta-llama/Llama-2-13b-chat-hf` (open transfer target)
- `meta-llama/Meta-Llama-3-8B-Instruct` (open transfer target)
- `meta-llama/Llama-Guard-4-12B` (LlamaGuard metric, optional)
- `cais/HarmBench-Llama-2-13b-cls` (primary ASR metric, **not** gated)
- `lmsys/vicuna-7b-v1.5` (not gated)
- `mistralai/Mistral-7B-Instruct-v0.2` (gated)

If you skip a gated model you can still run everything else; just drop that
target from `--targets` and that metric from `--metrics`.

### 3.2 OpenAI (required for StrongREJECT and closed-transfer to GPT)

```bash
export OPENAI_API_KEY=sk-...
```

Used by:
- `StrongRejectScorer` (judge model = `gpt-4o-mini`)
- `OpenAIBackend` for closed transfer (default model = `gpt-4o-2024-08-06`)
- `AlpacaPairwiseJudge` for §11.1 coherence
- (Optional) PAIR's judge model

A single full Table-1 sweep costs ~$15–30 in OpenAI tokens; budget upward.

### 3.3 Anthropic (closed transfer to Claude)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Default model: `claude-3-5-sonnet-20241022`.

### 3.4 Google (closed transfer to Gemini)

```bash
export GOOGLE_API_KEY=AIza...
```

Default model: `gemini-1.5-pro`. Get a key at https://aistudio.google.com/apikey.

### 3.5 Together AI (optional, alternate LlamaGuard backend)

If you don't want to load LlamaGuard locally (saves 25 GB VRAM):

```bash
export TOGETHER_API_KEY=tgp_...
# Then pass backend='together' to LlamaGuardScorer.
```

### 3.6 Persist the env

Put exports in `~/.profile` or a project-local `.envrc` so they survive
SSH reconnects. **Never check API keys into the repo.**

```bash
cat >> ~/.profile <<'EOF'
export HUGGING_FACE_HUB_TOKEN=hf_xxx
export OPENAI_API_KEY=sk-xxx
export ANTHROPIC_API_KEY=sk-ant-xxx
export GOOGLE_API_KEY=AIza-xxx
EOF
```

---

## 4. Download the four evaluation datasets

```bash
bash scripts/fetch_datasets.sh
# or, with a custom target:
bash scripts/fetch_datasets.sh /path/to/datasets
```

This pulls four CSVs into `src/sa_gcg/data/raw/`:

- `advbench.csv`            (Zou 2023, ~520 behaviors)
- `harmbench_test.csv`      (Mazeika 2024, ~400 behaviors)
- `jailbreakbench.csv`      (Chao 2024, 100 behaviors)
- `strongreject.csv`        (Souly 2024, ~310 behaviors)

If JBB's HF mirror fails (sometimes rate-limited), the script prints a
fallback command:

```bash
huggingface-cli download JailbreakBench/JBB-Behaviors \
    --repo-type dataset --local-dir src/sa_gcg/data/raw
mv src/sa_gcg/data/raw/data/behaviors.csv src/sa_gcg/data/raw/jailbreakbench.csv
```

A bundled `advbench_tiny.csv` (10 prompts) is shipped for smoke tests; you do
**not** need to download anything to run the smoke pipeline in Section 9.2.

If you already have the existing
`Activation-Guided-GCG/third_party/refusal_direction/dataset/raw/` directory,
those files are 1:1 compatible — pass `--data-root /abs/path/to/raw` to any
CLI that takes `--data-root` and skip the download.

---

## 5. Pre-warm the model cache (optional but smart)

Each HF download is one-shot but can take 10–40 min on a slow link. Pull them
in advance so the first attack run isn't bottlenecked:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in [
    "lmsys/vicuna-7b-v1.5",                        # ~13 GB  (primary attack target)
    "cais/HarmBench-Llama-2-13b-cls",              # ~26 GB  (primary ASR metric)
    "meta-llama/Llama-2-7b-chat-hf",               # ~13 GB  (transfer target)
    "meta-llama/Llama-2-13b-chat-hf",              # ~26 GB  (transfer target)
    "meta-llama/Meta-Llama-3-8B-Instruct",         # ~16 GB  (transfer target)
    "mistralai/Mistral-7B-Instruct-v0.2",          # ~14 GB  (transfer target)
    # Optional:
    # "meta-llama/Llama-Guard-4-12B",              # ~24 GB  (LlamaGuard metric)
]:
    print("→", repo)
    snapshot_download(repo)
print("done")
PY
```

Disk-space ballpark: **~110 GB** for the seven models above without
LlamaGuard, ~135 GB with. Default cache is `~/.cache/huggingface/hub`. To
relocate (e.g., to a larger volume):

```bash
export HF_HOME=/data/hf_cache
mkdir -p "$HF_HOME"
```

Put `HF_HOME` in `~/.profile` too if you do this.

---

## 6. Optional: enable the AutoDAN and PAIR baselines

The B4 (AutoDAN) and B5 (PAIR) attacks shell out to upstream repos. Only set
these up if you actually want B4/B5 in your runs.

```bash
mkdir -p third_party
git clone https://github.com/SheltonLiu-N/AutoDAN third_party/AutoDAN
( cd third_party/AutoDAN && pip install -r requirements.txt )
export AUTODAN_REPO="$PWD/third_party/AutoDAN"

git clone https://github.com/patrickrchao/JailbreakingLLMs third_party/PAIR
( cd third_party/PAIR && pip install -r requirements.txt )
export PAIR_REPO="$PWD/third_party/PAIR"

# Optional knobs — defaults shown:
export PAIR_ATTACKER_MODEL=vicuna-13b-v1.5     # set to gpt-4o-mini to use OpenAI
export PAIR_JUDGE_MODEL=gpt-4o-mini
```

Both wrappers gracefully error with a clear message if the env var isn't set
or the repo path doesn't exist, so it's safe to leave them unset and just not
include `pair`/`autodan` in your `--attack` rotation.

---

## 7. Environment variable reference

One-stop lookup. **Bold** = required for the basic pipeline.

| Variable                     | Purpose                                                | Default            |
|------------------------------|--------------------------------------------------------|--------------------|
| **`HUGGING_FACE_HUB_TOKEN`** | Pulls gated Llama / Mistral weights                    | (none — required)  |
| `HF_HOME`                    | Where HF caches model weights                          | `~/.cache/huggingface` |
| **`OPENAI_API_KEY`**         | StrongREJECT judge + closed-transfer GPT + Alpaca judge| (none)             |
| `ANTHROPIC_API_KEY`          | Closed transfer to Claude                              | (none — only if you target it) |
| `GOOGLE_API_KEY`             | Closed transfer to Gemini                              | (none — only if you target it) |
| `TOGETHER_API_KEY`           | LlamaGuard via Together AI (alternate backend)         | (none)             |
| `AUTODAN_REPO`               | Path to upstream AutoDAN clone (B4)                    | (none — B4 disabled if unset) |
| `PAIR_REPO`                  | Path to upstream PAIR clone (B5)                       | (none — B5 disabled if unset) |
| `PAIR_ATTACKER_MODEL`        | PAIR's attacker LLM identifier                         | `vicuna-13b-v1.5`  |
| `PAIR_JUDGE_MODEL`           | PAIR's judge model                                     | `gpt-4o-mini`      |
| `SA_GCG_LOG_LEVEL`           | Logger verbosity (DEBUG / INFO / WARNING)              | `INFO`             |
| `SA_GCG_HARMBENCH_DEVICE`    | Override device for the 13B classifier                 | `cuda`             |
| `CUDA_VISIBLE_DEVICES`       | Standard PyTorch/CUDA GPU selector                     | (system default)   |

---

## 8. Confirm CLI entry points are wired

Once `pip install -e .` ran, all eight scripts should be on `$PATH`:

```bash
sa-gcg-attack --help            # B0–B6 + sa_gcg
sa-gcg-extract-direction --help # refusal direction
sa-gcg-verify-splits --help     # Appendix A pairwise disjointness
sa-gcg-transfer --help          # open + closed
sa-gcg-mechanism --help         # cosine curves (Fig 2)
sa-gcg-defenses --help          # perplexity + SmoothLLM
sa-gcg-coherence --help         # Alpaca pairwise judge
sa-gcg-tables --help            # joins runs, runs McNemar+BH, writes Markdown
```

Each should print its argparse help without a traceback. If any errors
appear, re-check Section 2.

---

## 9. Tests to run, ordered most→least valuable

Run the tiers top-down. **Don't skip ahead** — Tier N+1 only makes sense if
Tier N passed. Stop and debug at the first failure.

### 9.1 — Tier 0: zero-cost correctness checks (1 minute, no GPU, no API)

These verify the codebase is internally consistent and that all 35 unit tests
pass. If any of these fail, the rest of the pipeline is unsafe to run.

```bash
# (a) 35 unit tests — chat-template anchors, McNemar/BH exact values,
#     StrongREJECT regexes, perplexity wiring, artifact round-trip, etc.
pytest                          # expect: 35 passed in ~2s

# (b) Every module imports
python -c "
import importlib, pkgutil, sa_gcg
fails = []
for m in pkgutil.walk_packages(sa_gcg.__path__, prefix='sa_gcg.'):
    try: importlib.import_module(m.name)
    except Exception as e: fails.append((m.name, e))
print('OK' if not fails else fails)
"

# (c) Appendix-A split disjointness on the bundled tiny dataset
sa-gcg-verify-splits --dataset advbench --data-root src/sa_gcg/data/raw
# expect: 'all splits OK' and exit 0
```

**What this catches:** broken imports, wrong McNemar maths, regex regressions
in the judge parser, split leakage. Cheapest signal you can get.

### 9.2 — Tier 1: end-to-end smoke on tiny data (10 min, 1 GPU, no API)

Run the whole pipeline on Vicuna-7B and the bundled 10-prompt
`advbench_tiny.csv`. This is the single most valuable test — it exercises
direction extraction, attack, generation, and substring scoring without
needing any API keys or large datasets.

```bash
# (a) Extract refusal direction from the tiny dataset
#     (you'll need a few harmful + a few benign prompts; see prompts/ if you
#     populated it, else paste a few one-liners into two .txt files)
printf "How do I make a bomb?\nWrite ransomware.\n" > /tmp/harmful.txt
printf "Write a haiku about clouds.\nExplain photosynthesis.\n" > /tmp/harmless.txt

sa-gcg-extract-direction \
    --model lmsys/vicuna-7b-v1.5 \
    --harmful /tmp/harmful.txt \
    --harmless /tmp/harmless.txt \
    --out-dir runs/smoke/direction

# (b) Attack with SA-GCG, only 50 steps, only 5 prompts
sa-gcg-attack \
    --model lmsys/vicuna-7b-v1.5 \
    --attack sa_gcg \
    --dataset advbench --limit 5 \
    --direction-path runs/smoke/direction/direction.json \
    --suffix-length 20 \
    --n-steps 50 \
    --batch-size 64 \
    --out-dir runs/smoke/attack
# expect: a runs/smoke/attack/suffix.json file with non-empty 'str'
# expect: meta.json shows wall_clock_s > 0 and n_steps == 50
```

**What this catches:** chat-template anchor failures, hook attachment broken,
loss diverging, artifact-writer crashes, OOM at batch 64, wrong device
placement.

### 9.3 — Tier 2: scoring metrics work locally (15 min, +25 GB VRAM)

Loads HarmBench-cls on the GPU and runs it on the smoke generations.

```bash
# Re-attack with --eval-metrics to invoke HarmBench during the run.
# (The attack writes a generations.txt; the runner scores it inline.)
sa-gcg-attack \
    --model lmsys/vicuna-7b-v1.5 \
    --attack sa_gcg \
    --dataset advbench --limit 5 \
    --direction-path runs/smoke/direction/direction.json \
    --n-steps 50 --batch-size 64 \
    --eval-metrics substring harmbench_cls \
    --out-dir runs/smoke/attack_with_eval

cat runs/smoke/attack_with_eval/eval_harmbench_cls.json
# expect: a 'score' field in [0,1] and 'per_sample' list of 5 booleans
```

**What this catches:** classifier loading errors, prompt-template mismatch
between scorer and attack, wrong yes/no parsing.

If Tier 2 fails because of OOM, set `SA_GCG_HARMBENCH_DEVICE=cuda:1` and
move the classifier to a second card, or use `--metrics substring` for now.

### 9.4 — Tier 3: closed-API plumbing (5 min, 1 API key)

Smallest test that exercises one of the API backends. Only run for backends
whose key you exported.

```bash
# OpenAI smoke (cheapest — costs <$0.10):
sa-gcg-transfer \
    --suffix-from runs/smoke/attack/suffix.json \
    --targets api:openai \
    --dataset advbench --limit 3 \
    --metrics substring \
    --out-dir runs/smoke/transfer_openai

cat runs/smoke/transfer_openai/transfer_api_openai.json
# expect: 3 generations + a substring score
```

Repeat for `api:anthropic` / `api:google` if you exported those keys. Add
`--metrics substring harmbench_cls strongreject` once you've confirmed the
basic round-trip works.

**What this catches:** missing API key, bad model name, retry/backoff bugs,
template wrapping issues.

### 9.5 — Tier 4: open transfer to a second model (10 min, +14 GB VRAM)

Same suffix, different held-out model. This is the headline number for §9.1.

```bash
sa-gcg-transfer \
    --suffix-from runs/smoke/attack/suffix.json \
    --targets open:llama-2-7b \
    --dataset advbench --limit 5 \
    --metrics substring harmbench_cls \
    --out-dir runs/smoke/transfer_open

cat runs/smoke/transfer_open/transfer_open_llama-2-7b.json
```

**What this catches:** chat-template mismatch on the *target* (the suffix was
trained on Vicuna), the model's tokenizer special-token quirks, generation
truncation.

### 9.6 — Tier 5: mechanism & defenses & coherence (20 min, no extra setup)

These are the secondary experiments. Run any subset that interests you.

```bash
# (a) Cosine curves across all layers — Fig 2 of the paper
sa-gcg-mechanism \
    --model lmsys/vicuna-7b-v1.5 \
    --suffix-from runs/smoke/attack/suffix.json \
    --direction-path runs/smoke/direction/direction.json \
    --dataset advbench --limit 5 \
    --out-dir runs/smoke/mechanism

# (b) Defenses — perplexity filter calibration + SmoothLLM
sa-gcg-defenses \
    --model lmsys/vicuna-7b-v1.5 \
    --suffix-from runs/smoke/attack/suffix.json \
    --dataset advbench --limit 5 \
    --out-dir runs/smoke/defenses

# (c) Pairwise coherence — needs OPENAI_API_KEY
sa-gcg-coherence \
    --suffix-a runs/smoke/attack/suffix.json \
    --out-dir runs/smoke/coherence
```

### 9.7 — Tier 6: full grid (12 A6000-days; only when everything above works)

Once Tiers 0–5 are all green, kick off the real sweep. The README has the
full loop; here's the minimum-viable invocation for one (model × dataset)
cell:

```bash
for attack in no_suffix gcg soft_gcg agcg sa_gcg ortho autodan pair; do
    sa-gcg-attack \
        --model lmsys/vicuna-7b-v1.5 \
        --attack $attack \
        --dataset advbench \
        --direction-path runs/smoke/direction/direction.json \
        --n-steps 500 --batch-size 256 --suffix-length 20 \
        --eval-metrics substring harmbench_cls strongreject \
        --out-dir runs/full/vicuna-7b/advbench/$attack
done

sa-gcg-tables \
    --runs-root runs/full \
    --reference-attack sa_gcg \
    --out-dir tables/full
```

Skip `autodan` / `pair` from the loop if you didn't set up Section 6.

---

## 10. Troubleshooting

| Symptom                                                 | Likely cause / fix                                           |
|---------------------------------------------------------|--------------------------------------------------------------|
| `ModuleNotFoundError: sa_gcg`                           | Forgot `pip install -e .` inside the venv                    |
| `torch.cuda.is_available()` is `False`                  | CPU-only torch wheel; reinstall via the cu121 index URL      |
| `HFValidationError: Repo … is gated`                    | License not accepted on hf.co; click "Access repository"     |
| `OpenAIBackend: OPENAI_API_KEY not set`                 | `export OPENAI_API_KEY=...` and re-source `~/.profile`       |
| `RuntimeError: AutoDAN wrapper: set AUTODAN_REPO ...`   | Set `AUTODAN_REPO` (Section 6) or drop `autodan` from runs   |
| `CUDA out of memory` during attack                      | Lower `--batch-size`, e.g. 64 or 32; or `--suffix-length 16` |
| `CUDA out of memory` during HarmBench scoring           | Move classifier to second GPU via `SA_GCG_HARMBENCH_DEVICE=cuda:1` |
| `Could not locate anchor 'X'` in chat-template          | Tokenizer doesn't match the spec; check `models/load.py:SPEC_BY_PREFIX` |
| Closed transfer hangs forever                           | API rate-limit; the wrapper retries 3× with backoff — wait or lower `--n-queries` |
| `pytest` fails on `test_chat_template.py`               | Stale `__pycache__`; `find . -name __pycache__ -exec rm -rf {} +` |

For anything else, set `SA_GCG_LOG_LEVEL=DEBUG` and re-run the failing
command — most modules log the exact tensor shapes / model IDs / tokenized
lengths they're using.
