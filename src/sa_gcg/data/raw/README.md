# Datasets — fetching instructions

The codebase expects these files in `src/sa_gcg/data/raw/` (or a path you pass
via `--data-root` / `data_root=`):

| File                      | Source                                                                 |
|---------------------------|------------------------------------------------------------------------|
| `advbench.csv`            | https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv |
| `harmbench_test.csv`      | https://github.com/centerforaisafety/HarmBench/raw/main/data/behavior_datasets/harmbench_behaviors_text_test.csv |
| `jailbreakbench.csv`      | `huggingface-cli download JailbreakBench/JBB-Behaviors --repo-type dataset` and use `data/behaviors.csv` |
| `strongreject.csv`        | https://github.com/dsbowen/strong_reject/raw/main/strong_reject/eval_files/strongreject_dataset.csv |

If you have the existing `Activation-Guided-GCG/third_party/refusal_direction/dataset/raw/`
directory, those files are 1:1 compatible — pass `--data-root
/path/to/refusal_direction/dataset/raw` and skip the downloads.

A convenience downloader is in `scripts/fetch_datasets.sh`.
