# SA-GCG: Soft Activation-Guided GCG

Composite-loss adversarial-suffix attack on safety-aligned chat LLMs.
Implementation of the eight attacks (B0-B6 + SA-GCG) and the full evaluation
pipeline described in `EVAL_PLAN.md` (NeurIPS 2026 submission).

> SA-GCG = Soft-GCG continuous relaxation (Gumbel-softmax over per-position
> logits) + Activation-Guided GCG residual-stream projection loss against
> the Arditi (2024) refusal direction, combined under a GradNorm-balanced
> composite, then optionally polished with discrete GCG.

## Repository layout

```
sa_gcg/
  src/sa_gcg/
    attack/        # B0-B6 + SA-GCG (sa_gcg.py is the headline)
    eval/          # substring, HarmBench-cls, StrongREJECT, LlamaGuard, Alpaca
    transfer/      # open (HF) + closed (OpenAI / Anthropic / Google) backends
    mechanism/     # per-layer cosine curves, layer sweep
    defenses/      # perplexity filter, SmoothLLM, adaptive perplexity attack
    stats/         # paired McNemar, BH FDR, clustered bootstrap, Wilson CI
    direction/     # Arditi diff-of-means refusal-direction extraction
    losses/        # CE, activation projection, GradNorm
    schedule/      # Slushy 3-phase Gumbel-softmax schedule
    models/        # chat templates + residual-reader hook + HF loader
    data/          # AdvBench / HarmBench / JailbreakBench / StrongREJECT
    cli/           # 8 entry-point scripts (see below)
    utils/         # artifact writer, logging, seed
  tests/           # 35 fast unit tests (no GPU, no API keys)
  scripts/         # fetch_datasets.sh
  configs/         # placeholder for experiment YAMLs
  prompts/         # judge prompts, harmful/harmless extraction prompts
  pyproject.toml
  environment.yml
  EVAL_PLAN.md     # (in repo root, one level up)
```

## Install

### Conda

```bash
conda env create -f environment.yml
conda activate sa_gcg
pip install -e .[api]            # adds openai/anthropic/google-generativeai
```

### Pip-only

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[api,dev]"
```

Python ≥ 3.10. Tested on 3.11 and 3.13.

## Datasets

Run the fetcher once:

```bash
bash scripts/fetch_datasets.sh
# writes advbench.csv, harmbench_test.csv, jailbreakbench.csv, strongreject.csv
# under src/sa_gcg/data/raw/
```

If you already have the AGCG / refusal_direction repo's `dataset/raw`,
pass `--data-root /path/to/that/raw` to any CLI and skip the downloads.

A 5-row `advbench_tiny.csv` is bundled for offline smoke testing.

For Arditi-style refusal-direction extraction, drop two newline-separated
text files (`harmful.txt`, `harmless.txt`) under `prompts/`. We
recommend the 128/128 split from
`tatsu-lab/alpaca` (harmless) and `walledai/AdvBench` (harmful).

## Quick start: reproduce the headline result on Llama-2-7B-Chat

The four-command happy path. Each step writes its artifact to `runs/`.

### 1. Extract the refusal direction

```bash
sa-gcg-extract-direction \
    --model meta-llama/Llama-2-7b-chat-hf \
    --harmful prompts/harmful128.txt \
    --harmless prompts/harmless128.txt \
    --out runs/direction/llama2.pt
```

This forwards each prompt through every decoder block, takes the
per-layer harmful-vs-harmless mean activation difference, and selects the
best `(layer, position)` by the Arditi screen (or the cheap mean-projection
heuristic when the bypass/induce/KL eval callables aren't wired yet).

### 2. Train the headline SA-GCG suffix (universal)

```bash
sa-gcg-attack \
    --attack sa_gcg \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dataset advbench --limit 25 \
    --universal --n-steps 500 \
    --activation-scope layer --composite-lambda 1.0 \
    --polish-steps 50 \
    --direction-path runs/direction/llama2.pt \
    --out-dir runs/sa_gcg/llama2/seed_0 \
    --seed 0
```

Repeat with `--seed 1..4` for the 5-seed mean ± SE in Tables 2/3.

### 3. Open transfer

```bash
sa-gcg-transfer \
    --suffix-from runs/sa_gcg/llama2/seed_0 \
    --dataset advbench --offset 25 --limit 100 \
    --targets open:vicuna-7b open:llama-2-13b open:llama-3-8b open:mistral-7b \
    --out-dir runs/sa_gcg/llama2/seed_0/transfer
```

### 4. Closed transfer (set API keys first)

```bash
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=... GOOGLE_API_KEY=...
sa-gcg-transfer \
    --suffix-from runs/sa_gcg/llama2/seed_0 \
    --dataset jailbreakbench --limit 100 \
    --targets api:openai api:anthropic api:google \
    --out-dir runs/sa_gcg/llama2/seed_0/transfer_closed \
    --n-queries 3
```

### 5. Make tables

```bash
sa-gcg-tables --runs-root runs --out-dir tables
# tables/raw_rows.csv, mcnemar.csv (BH-corrected), main_tables.md
```

## All baselines (B0-B6) in one loop

```bash
for attack in no_suffix gcg soft_gcg agcg sa_gcg ortho autodan pair; do
    sa-gcg-attack \
        --attack $attack \
        --universal --n-steps 500 \
        --dataset advbench --limit 25 \
        --direction-path runs/direction/llama2.pt \
        --out-dir runs/$attack/llama2/seed_0 \
        --seed 0
done
```

`autodan` / `pair` shell out to upstream repos. Set their paths:

```bash
git clone https://github.com/SheltonLiu-N/AutoDAN third_party/AutoDAN
git clone https://github.com/patrickrchao/JailbreakingLLMs third_party/PAIR
export AUTODAN_REPO=$PWD/third_party/AutoDAN
export PAIR_REPO=$PWD/third_party/PAIR
```

`ortho` (Arditi weight orthogonalization) returns an empty suffix and a
hook list — use the helper `defenses.ortho.directional_ablation_hooks`
in your eval if you want to compare against weight-edit "suffixes" at
generation time. The default eval just records the empty suffix; for full
parity with the published Ortho numbers register the hooks on the model
before generation.

## Mechanism (RQ3)

```bash
sa-gcg-mechanism \
    --suffix-from runs/sa_gcg/llama2/seed_0 \
    --direction-path runs/direction/llama2.pt \
    --dataset advbench --offset 25 --limit 100 \
    --out-dir runs/mechanism/sa_gcg
```

Random-direction control: re-run the attack with `--random-direction` and
re-eval. Layer sweep is in `sa_gcg.mechanism.layer_sweep` (no CLI yet
because it's an inner loop over the attack; call it from a notebook).

## Defenses (RQ6)

```bash
sa-gcg-defenses \
    --suffix-from runs/sa_gcg/llama2/seed_0 \
    --base-model meta-llama/Llama-2-7b-hf \
    --reference-corpus prompts/alpaca_responses.txt \
    --dataset advbench --offset 25 --limit 100 \
    --out-dir runs/defenses/sa_gcg
```

Adaptive: re-run `sa-gcg-attack ... --perplexity-penalty 0.1` and rescore.

## Coherence (§11.1)

```bash
sa-gcg-coherence \
    --suffix-from runs/sa_gcg/llama2/seed_0 \
    --alpaca-prompts prompts/alpaca_benign_100.txt \
    --judge-model gpt-4o-mini \
    --out-dir runs/coherence/sa_gcg
```

## Tests

```bash
pytest                          # 35 fast unit tests (~2 s on CPU)
pytest -k "not slow"            # explicitly skip GPU/API tests
```

The fast tests verify:
- substring / StrongREJECT parser correctness
- McNemar exact p-value, BH FDR, clustered-bootstrap CI
- chat-template anchor span detection (Llama-2/3, Vicuna, Mistral)
- slushy / linear schedule shapes
- artifact writer round-trip
- registry of all 8 attacks
- SmoothLLM character perturbations
- perplexity-filter wiring (mocked)
- data registry + split disjointness

GPU/API-marked tests (`@pytest.mark.slow`, `@pytest.mark.api`) require
the heavy classifier model loads and live API keys; skip them in CI.

## File-by-file tour

| File | Role |
|------|------|
| `attack/sa_gcg.py` | Headline algorithm (Appendix E pseudocode) |
| `attack/gcg.py` | Faithful Zou-2023 reimplementation |
| `attack/soft_gcg.py` | SA-GCG with `composite_lambda=0` (S4) |
| `attack/activation_gcg.py` | Discrete GCG over the activation loss only |
| `attack/ortho.py` | Arditi weight orthogonalization (returns empty + hooks) |
| `attack/autodan_wrapper.py` / `pair_wrapper.py` | shell-out to upstream repos |
| `losses/activation_loss.py` | `single` / `layer` / `global` scope projection loss |
| `losses/gradnorm.py` | Single-pass GradNorm freeze at step 1 |
| `losses/ce_loss.py` | First-token-weighted target CE |
| `schedule/slushy.py` | 3-phase Gumbel-softmax temperature |
| `direction/extractor.py` | Diff-of-means + Arditi screen |
| `models/chat_template.py` | Per-family templates with anchor-based suffix span |
| `models/hooks.py` | `ResidualReader` forward-pre-hook |
| `eval/runner.py` | Generate once, score with N metrics |
| `eval/harmbench_cls.py` | `cais/HarmBench-Llama-2-13b-cls` (primary) |
| `eval/strongreject.py` | Souly et al. 2024 rubric judge |
| `eval/alpaca_judge.py` | A/B coherence judge |
| `transfer/closed_api.py` | OpenAI / Anthropic / Google backends |
| `mechanism/cosine_curves.py` | Per-layer cosine to refusal direction |
| `defenses/perplexity.py` | Calibrated 99th-percentile NLL filter |
| `defenses/smoothllm.py` | Robey 2023 char-perturb majority-vote |
| `defenses/adaptive.py` | Two-model perplexity proxy for adaptive SA-GCG |
| `stats/tests.py` | McNemar / BH / clustered bootstrap / Wilson |
| `cli/run_attack.py` | `sa-gcg-attack` |
| `cli/extract_direction.py` | `sa-gcg-extract-direction` |
| `cli/verify_splits.py` | `sa-gcg-verify-splits` (Appendix A) |
| `cli/run_transfer.py` | `sa-gcg-transfer` (open + closed) |
| `cli/run_mechanism.py` | `sa-gcg-mechanism` (Figure 2) |
| `cli/run_defenses.py` | `sa-gcg-defenses` (perplexity + SmoothLLM) |
| `cli/run_coherence.py` | `sa-gcg-coherence` (§11.1) |
| `cli/make_tables.py` | `sa-gcg-tables` (joins runs into Tables + McNemar) |

## Compute envelope

EVAL_PLAN budgets the full experiment at ~290 GPU-hr (12 A6000-days). Run
the headline cell first (~5 hr/seed × 5 seeds = 25 GPU-hr) and use the
remainder for baselines / transfer / mechanism. See EVAL_PLAN §15 for
the full breakdown and the priority list when compute is tight.

## Reproducibility

- Seeds 0-4 for universal runs; 0-2 for individual.
- All artifacts (suffix, loss curve, cosine curve, generations, classifier
  labels) dump to `runs/{attack}/{model}/seed_{k}/` for downstream
  re-aggregation.
- API runs log model version, system fingerprint, and date inside
  `meta.json`.
- `torch.use_deterministic_algorithms(True)` would break flash-attention;
  we accept non-bitwise determinism and let cross-seed variance capture
  residual noise.

## Citation

```
@unpublished{sa_gcg_2026,
  author = {SA-GCG authors},
  title  = {Soft Activation-Guided GCG: composite-loss adversarial suffix attacks},
  year   = {2026},
  note   = {NeurIPS submission. See EVAL_PLAN.md for the experimental design.}
}
```

## License

MIT. Released for the purpose of red-teaming research per the responsible
disclosure guidelines of Zou et al. (2023) and Arditi et al. (2024). Do
not use the trained suffixes against real production systems without the
operator's permission.
