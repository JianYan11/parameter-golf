# Parameter Golf — contestant toolkit (community fork)

Language models are just fancy autocomplete: you predict the next token, then the next, then the next. Parameter Golf asks you to do that *really well* under a brutal budget — and honestly, half the fun is watching your little transformer's "thought process" spill out as text. **[Official challenge, leaderboard, FAQ, grants, Discord, full setup → openai/parameter-golf](https://github.com/openai/parameter-golf).** Everything authoritative lives there. **This repo is not OpenAI** — it's **[github.com/JianYan11/parameter-golf](https://github.com/JianYan11/parameter-golf)**, a fork with a small set of tools so your R&D loop feels less like bureaucracy and more like *building*.

---

https://github.com/user-attachments/assets/0229009b-9495-4bcf-a3c6-b8394bb42ffa



## See it spit tokens (CLI continuation)

Grab a checkpoint, point `generate_demo.py` at the FineWeb tokenizer, and you get a **terminal-native** streaming session — the same mental model as training (autoregressive decode), except now you can *feel* whether your plumbing works before you've burned a GPU week. Here's what that looks like in practice:

<video controls playsinline width="100%" style="max-width: 720px;" src="https://raw.githubusercontent.com/JianYan11/parameter-golf/main/docs/assets/cli-continuation-demo.mp4">
  Video not loading? Open the file on GitHub: <a href="https://github.com/JianYan11/parameter-golf/blob/main/docs/assets/cli-continuation-demo.mp4">cli-continuation-demo.mp4</a> — or use the copy under <code>docs/assets/</code> after you clone.
</video>

*The clip above is served from `docs/assets/cli-continuation-demo.mp4` on the default branch; if you fork the repo, swap the URLs above for your GitHub username/org, or just open the file locally.*

---

## What you get here (why it's more fun *and* faster)

Parameter Golf is already addictive: tight caps, clever records, everyone chasing `val_bpb`. The friction is the *stuff around* training — wiring checkpoints, eyeballing runs, remembering what you changed last Tuesday. This fork tries to strip that away.

| Piece | Why it matters |
|-------|----------------|
| [`scripts/download_cli_demo_checkpoint.py`](scripts/download_cli_demo_checkpoint.py) | **Instant artifact-shaped checkpoint** — no training required. Random init, so it's for wiring tests only, but that's the point: *fail fast* before the expensive part. |
| [`scripts/generate_demo.py`](scripts/generate_demo.py) | **Streaming decode** from real `train_gpt.py` checkpoints (int8 round-trip, DDP prefixes, etc.). Uses **`GPT.forward_logits()`** in this fork's [`train_gpt.py`](train_gpt.py). Optional FineWeb val snippet if you pass `--data-path` + tokenizer — nice little reality check on data + tokenizer + model agreeing. |
| [`train_gpt.py`](train_gpt.py) → **`results.tsv`** | Every run appends a row on **rank 0**: **`COMPLETE`** with round-trip **`val_bpb`**, peak VRAM, short git hash — or **`CRASH`** when the universe says no. Tag runs with **`EXPERIMENT_DESC`**. The file is **gitignored** so your lab notebook stays local. |
| [`analysis.ipynb`](analysis.ipynb) | Turns that TSV into pictures and tables. Humans are surprisingly bad at mentally plotting twenty runs; matplotlib is not. |
| [`scripts/h100_time_guess.py`](scripts/h100_time_guess.py) | **Napkin math** vs "did I accidentally invent a three-hour training run?" — sanity against the **600 s / 8×H100** *story* of the official track (still: truth is always measured on real iron). |
| [`agent.md`](agent.md) | The long-form playbook: env vars, greps, hardware notes, and the **autoresearch** loop below. |

**`records/` PRs** must still obey [OpenAI's submission rules](https://github.com/openai/parameter-golf). Treat the logging + demos here as **dev ergonomics**; ship a self-contained `train_gpt.py` in your record when you go official.

---

## Autoresearch: turn training into a loop agents (and humans) can run

The heart of this fork's *workflow* doc is **[`agent.md`](agent.md) §7 — Autonomous research program**. Think of it as "how to iterate without losing the plot." Roughly:

1. **Session setup** — Agree on a tag, cut a branch like `research/<tag>`, **fetch upstream** so you're not accidentally fork-diverging from challenge reality, skim README + `data/README.md` + `train_gpt.py`, confirm FineWeb + tokenizer on disk. `results.tsv` needs no manual header; the first successful run creates it.

2. **Hard guardrails (what you CAN vs CANNOT touch)** — Primary edit surface is **[`train_gpt.py`](train_gpt.py)** subject to artifact size, integrity, and official time rules. You don't get to cheat `val_bpb`, smuggle unpaid val into the artifact, or quietly break comparability — [FAQ ON THE OFFICIAL README](https://github.com/openai/parameter-golf) still wins.

3. **Simplicity criterion** — Lower `val_bpb` is the game, but *complexity has a tax*. A tiny gain that adds a pile of hacks probably isn't worth it; **deleting** code and matching or beating loss is a moral victory. [`agent.md`](agent.md) spells out that tradeoff explicitly.

4. **Literature + one-line hypothesis** — Before each iteration, you're encouraged to **search** (papers, blogs, docs), form a **single testable sentence**, and stash citations in **`EXPERIMENT_DESC`** / the TSV row so the idea trace is *auditable* later. Science, but fast and messy in the good way.

5. **The experiment loop** — Mine **this repo's own [`records/`](records/)** (not just Google): read `submission.json`, READMEs, logs; propose **three meaningfully different mechanistic directions**; human picks A/B/C. Then: **one focused change** → commit → **`torchrun ... > run.log 2>&1`** (no drowning the terminal in `tee`) → grep for `final_int8_zlib_roundtrip` and submission size → **keep or revert** based on round-trip `val_bpb` and whether you're still under **16 000 000** bytes. Rinse. Repeat. The loop is designed so an autonomous agent can **keep going** once setup is done — with a deliberate **human checkpoint** when picking the next research route.

6. **Timeouts & crashes** — Wall-clock expectations, `CRASH` rows, and **when to stop asking and keep iterating** (vs when to pivot) are all spelled out in §7.5–§7.6.

If you only read one extra file after this README, make it **`agent.md`**: it's the missing manual for **structured, high-throughput** Parameter Golf hacking.

---

## Clone & track upstream

```bash
git clone https://github.com/JianYan11/parameter-golf.git
cd parameter-golf
```

```bash
git remote add upstream https://github.com/openai/parameter-golf.git   # if missing
git fetch upstream
git log upstream/main..HEAD --oneline   # commits only on this fork
```

---

## Minimal environment & data

Python / CUDA / MLX setup: **[upstream Getting Started](https://github.com/openai/parameter-golf#getting-started)** and [`requirements.txt`](requirements.txt). Layout for shards + tokenizer: **[data/README.md](data/README.md)**.

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1   # more shards for real training
```

```bash
RUN_ID=try1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

(Fork adds TSV logging automatically unless you set `DISABLE_RESULTS_TSV=1`.)

---

## Quick commands (fork-specific)

**Demo checkpoint (no training)**

```bash
python scripts/download_cli_demo_checkpoint.py -o cli_demo_checkpoint.pt
```

**Stream generation** (tokenizer required; add `--data-path ./data/datasets/fineweb10B_sp1024/` for a short val decode first)

```bash
python scripts/generate_demo.py \
  --checkpoint cli_demo_checkpoint.pt \
  --tokenizer ./data/tokenizers/fineweb_1024_bpe.model \
  --no-show-sample
```

**Tag `results.tsv` rows**

```bash
EXPERIMENT_DESC="idea_v3_muon_lr + https://arxiv.org/abs/...." torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Then open **`analysis.ipynb`**.

**Time ballpark**

```bash
python scripts/h100_time_guess.py
python scripts/h100_time_guess.py check run.log
```

---

## More

- **Challenge authority:** [openai/parameter-golf](https://github.com/openai/parameter-golf)  
- **Autoresearch detail:** [`agent.md`](agent.md)  
- **Attribution:** [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
