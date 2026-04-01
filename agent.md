# Parameter Golf — local research program

Operational guide for autonomous iteration on this repo: challenge goals, hard limits, paths, commands, log parsing, and an experiment loop. **This document assumes local training on 2× RTX 5090 with a 30-minute wall-clock cap** (`MAX_WALLCLOCK_SECONDS=1800`); official leaderboard limits stay 8× H100 and 600 s (see §2). The **autoresearch-style workflow** (setup, CAN/CANNOT, loop) is in **§7**. Authoritative rules and FAQs remain in [README.md](README.md) and [data/README.md](data/README.md).

## 1. Goals and metrics

- **Objective**: Train the best language model under the challenge caps, scored on **FineWeb validation** using a **tokenizer-agnostic** metric: **`val_bpb` (bits per byte)**. **Lower is better.**

## 2. Hard constraints

| Constraint | Official track (leaderboard) | Notes |
|------------|-------------------------------|--------|
| Training time | ≤ **10 minutes** wall clock on **8× H100 SXM** | Official leaderboard track. |
| Training cap in code | `MAX_WALLCLOCK_SECONDS` (default **600** in [`train_gpt.py`](train_gpt.py) `Hyperparameters`) | This lab uses **1800** (30 min) on 2×5090 for iteration; use **600** when rehearsing an official 10-minute submission. Set `MAX_WALLCLOCK_SECONDS=0` to disable the cap (not valid for official submissions). |
| Artifact | **Code bytes + zlib-compressed int8 model** ≤ **16,000,000** bytes (decimal **16 MB**) | Counted code is expected to live in **`train_gpt.py`** per challenge FAQ. |
| Evaluation time | ≤ **10 minutes** on 8×H100 for evaluation (in addition to training) | See README FAQ. |
| Integrity | No sneaking validation into training or into the artifact as unpaid “prefix”; TTT only on tokens already evaluated | See README FAQ on test-time training. |

**This lab (local iteration)** — not the official hardware budget:

- **GPUs**: **2× NVIDIA RTX 5090**
- **Training wall clock**: **30 minutes** per run → **`MAX_WALLCLOCK_SECONDS=1800`**
- **Processes**: `torchrun --standalone --nproc_per_node=2` (one process per GPU)

For an official **records/** submission, reproduce under **8× H100** and **600 s** (and README eval time limits). When optimizing for the official track after iterating on 5090, expect to **re-tune** steps or schedule on 8×H100 so training (and eval) stay within caps while maximizing quality.

## 3. Environment

From the repository root, activate the project venv: `source .venv/bin/activate`. If `.venv` is missing, create it with `python3 -m venv .venv` and install deps per [README.md](README.md). Do not use imports to hide extra parameters or unfair compute.

## 4. Repository paths (canonical layout)

| Purpose | Path |
|---------|------|
| Cached FineWeb + tokenizer download | `python3 data/cached_challenge_fineweb.py --variant sp1024` (optional: `--train-shards N`) |
| Training shards | `./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin` |
| Validation shards | `./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin` |
| SentencePiece model | `./data/tokenizers/fineweb_1024_bpe.model` |
| Default training script (CUDA) | [`train_gpt.py`](train_gpt.py) |
| Record submissions | `records/track_10min_16mb/...` and `records/track_non_record_16mb/...` |

More detail: [data/README.md](data/README.md).

## 5. Standard run commands

After data is present, use **this lab’s** 2×5090 setup (**30 min** cap, **two** processes):

```bash
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=1800 \
torchrun --standalone --nproc_per_node=2 train_gpt.py
```

For an **official 10-minute / 8×H100** rehearsal, set `MAX_WALLCLOCK_SECONDS=600` and `nproc_per_node=8` on that machine. On a single GPU, use `nproc_per_node=1` and the same env vars.

**Frequently used environment variables** (see `Hyperparameters` in [`train_gpt.py`](train_gpt.py)):

- **Data / run**: `DATA_PATH`, `TOKENIZER_PATH`, `RUN_ID`, `SEED`
- **Budget / schedule**: `MAX_WALLCLOCK_SECONDS`, `ITERATIONS`, `WARMUP_STEPS`, `WARMDOWN_ITERS`, `TRAIN_BATCH_TOKENS`, `TRAIN_SEQ_LEN`
- **Logging / eval cadence**: `TRAIN_LOG_EVERY`, `VAL_LOSS_EVERY`, `VAL_BATCH_SIZE`
- **Model shape**: `VOCAB_SIZE`, `NUM_LAYERS`, `MODEL_DIM`, `NUM_HEADS`, `NUM_KV_HEADS`, `MLP_MULT`, `TIE_EMBEDDINGS`, `ROPE_BASE`, `LOGIT_SOFTCAP`
- **Optimizer (subset)**: `EMBED_LR`, `MATRIX_LR`, `HEAD_LR`, `MUON_MOMENTUM`, `GRAD_CLIP_NORM`

Redirect logs for unattended runs (**no `tee`** — avoid flooding agent context):

```bash
RUN_ID=my_experiment \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=1800 \
torchrun --standalone --nproc_per_node=2 train_gpt.py > run.log 2>&1
```

### 5.1 Development hardware (2× RTX 5090) vs official (8× H100)

**Commands**

- **This lab:** `torchrun --standalone --nproc_per_node=2 train_gpt.py` with `MAX_WALLCLOCK_SECONDS=1800`.
- **Official / replay:** `nproc_per_node=8` on an 8× H100 box; `MAX_WALLCLOCK_SECONDS=600` for a 10-minute submission rehearsal. Use the **same** env vars (besides cap and process count) for apples-to-apples **recipe** comparisons where possible.

**Why the training *recipe* can match across 2 and 8 GPUs**

In [`train_gpt.py`](train_gpt.py), `WORLD_SIZE` must divide 8, and `grad_accum_steps = 8 // world_size`. With `DistributedTokenLoader`, **per optimizer step the global token budget from `TRAIN_BATCH_TOKENS` is preserved** when switching between e.g. 2 and 8 ranks (micro-batch and gradient accumulation adjust together). So **5090 vs H100 is mainly wall-clock / throughput**, not a different global batch story unless you change env vars.

**No formal guarantee without an 8×H100 run**

Step time depends on kernels, NCCL, power/thermals, and model shape. If a config runs on 2×5090, it will **usually** run on 8×H100, but **only a real 8×H100 run** proves wall clock ≤ **600 s** with `MAX_WALLCLOCK_SECONDS=600` and your chosen `ITERATIONS` / schedule. For typical data-parallel workloads, **8× H100 is often faster per step** than 2×5090, so completing **N steps** under the local cap does **not** automatically mean you will **fail** the official cap for the same **N**—you may need **more steps** on H100 to use the full 10 minutes and maximize **`val_bpb`**. Do a **final smoke or calibration on target SKU** before submission.

**Rough wall-clock equivalence — use your logs, not a magic factor**

- Compare **`step_avg`** (ms per step) from training logs on **2×5090** vs a short run on **8×H100** when available: \( \text{time for } N \text{ steps} \approx N \times \text{step\_avg} \) on each stack.
- **Order-of-magnitude only:** for many Transformer setups, 8× H100 vs 2×5090 can differ by **several× to about an order of magnitude** in wall-clock **per training step** depending on width, sequence length, and kernels—so **~10 min official training** *might* correspond to **tens of minutes to ~1–2 h** on 2×5090 for the **same step count**, unless your measured `step_avg` says otherwise.

**Performance posture**

- Optimize for **lower** `final_int8_zlib_roundtrip` **`val_bpb`** subject to **`Total submission size int8+zlib`** ≤ **16 000 000**.
- **Eval** on the official track also has a **10 min** budget on 8×H100 (README FAQ); heavy custom eval must respect that on target hardware.

## 6. Logs, metrics, and artifact size lines

During training, validation lines include `val_bpb`. After training, the script quantizes, zlib-compresses, round-trips weights, and logs:

- **`final_int8_zlib_roundtrip`** — primary line for **val_loss** and **val_bpb** after int8+zlib round-trip.
- **`final_int8_zlib_roundtrip_exact`** — higher-precision duplicate for the same quantities.

**Submission-sized artifact** (what counts toward 16 MB alongside code) appears as:

- `Serialized model int8+zlib: … bytes`
- `Total submission size int8+zlib: … bytes`

**Suggested greps:**

```bash
grep 'final_int8_zlib_roundtrip' run.log
grep 'Total submission size int8+zlib' run.log
```

For step-wise validation (when `VAL_LOSS_EVERY` is enabled):

```bash
grep 'val_bpb:' run.log
```

## 7. Autonomous research program

### 7.1 Setup (per session / “new day”)

Work with the human once, then run the loop (§7.5):

1. **Agree on a run tag** (e.g. `mar28` or `2026-03-28`). Branch **`research/<tag>`** must **not** already exist for a fresh run, or use **`research/<tag>-gpu0`** for a parallel GPU track. (Prefix `research/` avoids clashing with upstream naming.)
2. **Create the branch:** `git fetch origin && git checkout main && git pull` (if applicable), then `git checkout -b research/<tag>`.
3. **Read in-scope files** before editing:
   - [README.md](README.md) — challenge rules, FAQ, integrity.
   - [data/README.md](data/README.md) — data layout, tokenizer, export notes.
   - [`train_gpt.py`](train_gpt.py) — `Hyperparameters`, model, training loop, eval, int8+zlib serialization.
   - [program.md](program.md) — this guide.
4. **Verify data:** `fineweb_train_*.bin`, `fineweb_val_*.bin`, and tokenizer at `TOKENIZER_PATH`; if missing, run `python3 data/cached_challenge_fineweb.py --variant sp1024` (see §4).
5. **Initialize `results.tsv`** with the header row only (see §7.5). The baseline row is filled after the first run.
6. **Confirm** with the human that setup looks good, then enter the experiment loop.

### 7.2 What you CAN / CANNOT do

**CAN (primary edit surface)**

- Change **[`train_gpt.py`](train_gpt.py)** — architecture, optimizers, schedules, quantization-related logic, logging cadence — subject to **§2** (artifact, integrity, official time caps for submissions).

**CAN (supporting)**

- Drive training via **environment variables** documented in `Hyperparameters` / §5.
- Add **optional dependencies** in the **spirit of README** (e.g. FlashAttention): record in a local `requirements.txt` or in a future `records/...` submission per **§8**.

**CANNOT**

- Violate **integrity / leaderboard rules** in [README FAQ](README.md): no unpaid validation “prefix” in the artifact; no sneaking validation into training; **TTT** only on validation tokens **already evaluated**; no network during evaluation as required by the FAQ.
- Bypass the **artifact cap** (code + int8+zlib ≤ **16 000 000** bytes) or **official** training/eval time caps when claiming a **records/** result.
- **Redefine or game `val_bpb`** (change eval to non-comparable semantics) without rigorous justification and **§8**-style disclosure.
- For **routine iteration**, treat **`data/`** pipeline scripts as **read-only** (shard format, tokenizer contract, download scripts) unless the human explicitly scopes a tokenizer/dataset experiment and accepts **§8** obligations.

### 7.3 Simplicity criterion

All else being equal, **simpler is better**. A small improvement in **`val_bpb`** (lower is better) that adds ugly complexity is **not** worth it. Conversely, **removing** something and getting equal or **better** **`val_bpb`** is a simplification win. When deciding whether to keep a change, weigh **complexity cost** against **improvement size**. A **0.001** better **`val_bpb`** that adds **20 lines** of hacky code? Probably not worth it. A **0.001** improvement **from deleting** code? **Definitely keep**. An improvement of **~0** **`val_bpb`** but **much simpler** code? **Keep**.

### 7.4 Literature and ideation (web search)

Before or during each iteration, **use web search / paper sources** (e.g. arXiv, blogs, docs) to find methods relevant to the current bottleneck (architecture, optimization, quantization, throughput under hardware caps, etc.). Form a **one-sentence, testable hypothesis**. **Cite** sources (title + URL) in the **`results.tsv` description** (preferred) so the trace of ideas is auditable.

### 7.5 Experiment loop

Work lives on **`research/<tag>`** (or the branch agreed in §7.1).

**Logging results (`results.tsv`)**

When an experiment finishes, append a row (tab-separated, **not** CSV — commas break in descriptions):

```text
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, **7** chars)
2. **`val_bpb`** from **`final_int8_zlib_roundtrip`** — use **0.000000** for crashes
3. peak VRAM **GB**, **.1f** (from logs if available; else **0.0**)
4. **status:** `keep` / `discard` / `crash`
5. short description of the change (**include paper/link hints** per §7.4 when applicable)

Example:

```text
commit	val_bpb	memory_gb	status	description
a1b2c3d	1.234500	44.0	keep	baseline
b2c3d4e	1.230000	44.2	keep	lower MATRIX_LR; Ref: https://arxiv.org/...
```

**Do not commit `results.tsv`** to git (keep it untracked or gitignored locally).

**LOOP** (repeat until manually stopped):

1. Note **branch** and **short commit**.
2. **Research / ideation** (§7.4).
3. Implement **one** focused change in **`train_gpt.py`** (or env-only change if appropriate).
4. `git commit`.
5. Run training with full log redirect, e.g. **`torchrun ... train_gpt.py > run.log 2>&1`** as in §5 — **no `tee`**, do not dump huge logs into agent context.
6. Parse logs (same greps as **§6**): **`grep 'final_int8_zlib_roundtrip' run.log`** and **`grep 'Total submission size int8+zlib' run.log`**; optionally **`grep 'val_bpb:' run.log`** if `VAL_LOSS_EVERY` is set. If those lines are missing, **`tail -n 50 run.log`** and treat as crash or mis-run.
7. Append **`results.tsv`**. **VRAM** is a **soft** constraint: some increase is acceptable for meaningful **`val_bpb`** gains, but avoid runaway memory.
8. **Advance or revert:** if **round-trip `val_bpb` improves** (lower), **`Total submission size int8+zlib`** is under **16 000 000**, and §7.3 accepts the complexity — **keep** the commit. Otherwise **discard** (e.g. `git checkout -- train_gpt.py` or reset). If **`val_bpb`** is flat or worse, revert rather than stacking noise.

**First run in a session:** always establish a **baseline** (stock or agreed starting `train_gpt.py`) before comparing ideas.

### 7.6 Timeouts, crashes, agent mode

- **Timeouts:** This lab expects ~**30  minutes** training (`MAX_WALLCLOCK_SECONDS=1800`) plus eval/quantization. If wall clock goes far beyond ~**2×** that budget or the process hangs, kill it, inspect **`tail -n 50 run.log`**, revert or fix.
- **Crashes:** trivial fixes (typo, import) → fix and re-run. Fundamentally broken idea → log **`crash`** in **`results.tsv`**, move on.
- **Agent mode:** After §7.1 setup, **do not** stop to ask whether to continue. Keep iterating until **manually** interrupted. If stuck, **search again** (§7.4), re-read **`train_gpt.py`** / README, combine prior near-misses, or try bolder architectural changes.

## 8. Submission checklist (abbreviated)

For **new SOTA** records (see [README.md — Submission Process](README.md)):

1. Beat existing SOTA by at least **0.005 nats** with evidence (e.g. **p < 0.01** across multiple seeds/logs unless the change is pure systems speed with unchanged ML).
2. If you change **tokenizer or dataset**, provide a rigorous argument that **`val_bpb`** is computed correctly.
3. Reproduce training in **under 10 minutes on 8×H100**.
4. PR adds only a new folder under the appropriate **`records/...`** track with at least: **`README.md`**, **`submission.json`**, **training log(s)**, and a runnable **`train_gpt.py`** .

---

Details, leaderboard policy, and community links: **[README.md](README.md)**. Data export and tokenizer rebuild: **[data/README.md](data/README.md)**.