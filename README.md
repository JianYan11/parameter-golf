# parameter-golf (community fork)

**Train smarter, iterate faster, and actually *see* your model sooner.** This repo is a community fork of [OpenAI’s Parameter Golf](https://github.com/openai/parameter-golf). You keep the same training stack and `records/` layout as upstream, but you get a small **contestant toolkit** designed to make local R&D **more efficient** (less time burned on setup and bookkeeping) and **more fun** (instant demos, readable experiment history, quick sanity checks against the 10-minute / 8×H100 mental model).

**Not affiliated with OpenAI.** Official rules, compute grants, Discord, and submission requirements live in [openai/parameter-golf](https://github.com/openai/parameter-golf).

## Why this fork feels better to train in

| Theme | What you get |
|--------|----------------|
| **Faster feedback** | Grab a **demo checkpoint without a full training run** and wire up streaming generation in minutes—not after hours of GPU time. |
| **More engaging loops** | **`generate_demo.py`** turns weights into **live text** so architecture and training changes feel tangible, not just loss printouts. |
| **Efficient experiment memory** | **`results.tsv` autologging** on rank 0 captures **COMPLETE** runs (round-trip **`val_bpb`**, peak VRAM, git hash, your **`EXPERIMENT_DESC`**) and **CRASH** lines when things blow up—no spreadsheet copy-paste. |
| **Clearer progress over time** | **`analysis.ipynb`** plots and summarizes many runs from that TSV so sweeps and lucky seeds are easy to compare. |
| **Saner time budgeting** | **`h100_time_guess.py`** gives napkin math and optional **`run.log`** checks against the **600s** training cap mindset—so you spend fewer surprises on wrong hardware assumptions. |

Backed by a longer local playbook in [`agent.md`](agent.md) (upstream sync, env vars, log patterns, experiment discipline).

## What we add vs upstream

| Addition | Purpose |
|----------|---------|
| [`scripts/download_cli_demo_checkpoint.py`](scripts/download_cli_demo_checkpoint.py) | Materialize an **FP16** checkpoint in **train_gpt.py shape** for wiring tests. Uses bundled [`demo_assets/baseline_demo_fp16.pt.lzma`](demo_assets/baseline_demo_fp16.pt.lzma) when present, or URL / `PARAMETER_GOLF_DEMO_CKPT_URL` / `PARAMETER_GOLF_DEMO_RAW_BASE`. Random init: **quality-free**, **pipeline-verification only**. |
| [`scripts/generate_demo.py`](scripts/generate_demo.py) | **Stream text** from checkpoints compatible with training (`final_model.pt`, int8 round-trip, DDP `module.*`, …). Needs **`GPT.forward_logits()`** in this fork’s [`train_gpt.py`](train_gpt.py). Optional FineWeb val preview with `--data-path` + tokenizer. |
| **`train_gpt.py` → `results.tsv`** | Rank 0 appends one row per run. Tunables: `EXPERIMENT_DESC`, `RESULTS_TSV_PATH`, `DISABLE_RESULTS_TSV=1`. File is **gitignored**—for local analysis only. |
| [`analysis.ipynb`](analysis.ipynb) | Read **`results.tsv`** and visualize aggregates (e.g. **`val_bpb`**). |
| [`scripts/h100_time_guess.py`](scripts/h100_time_guess.py) | Rough **8×H100 / 10 min** comparisons or step-based checks vs **`run.log`** (always validate on real iron for submissions). |
| [`agent.md`](agent.md) | Deep-dive workflow for day-to-day Parameter Golf hacking. |

**`records/` submissions:** Challenge rules still expect a self-contained **`train_gpt.py`** per entry. Treat logging and demos as **development affordances**; trim or justify extras before a minimal, byte-conscious record PR if you need to stay identical to a bare upstream record.

## Quickstart (clone *this* fork for `demo_assets/`)

**1) Demo checkpoint only (no training)**

```bash
python scripts/download_cli_demo_checkpoint.py -o cli_demo_checkpoint.pt
```

**2) Stream generation** (needs the SentencePiece tokenizer from your data setup—see [`data/README.md`](data/README.md))

```bash
python scripts/generate_demo.py \
  --checkpoint cli_demo_checkpoint.pt \
  --tokenizer ./data/tokenizers/fineweb_1024_bpe.model \
  --no-show-sample
```

Add `--data-path ./data/datasets/fineweb10B_sp1024/` for a short FineWeb **validation** decode before generation.

**3) Train with automatic experiment rows**

```bash
EXPERIMENT_DESC="muon_lr_sweep_A" torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Then open [`analysis.ipynb`](analysis.ipynb).

**4) Time ballpark + log check**

```bash
python scripts/h100_time_guess.py
python scripts/h100_time_guess.py check run.log
```

## Data & environment

- Dataset and tokenizer layout: [`data/README.md`](data/README.md).
- CUDA deps: upstream **`requirements.txt`**; MLX path: **`train_gpt_mlx.py`** and upstream docs.

## Stay aligned with OpenAI upstream

```bash
git remote add upstream https://github.com/openai/parameter-golf.git  # if missing
git fetch upstream
git log upstream/main..HEAD --oneline
```

## Acknowledgements

Challenge and code trace to OpenAI and `records/` contributors. Third-party notices: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
