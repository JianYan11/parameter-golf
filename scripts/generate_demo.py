#!/usr/bin/env python3
"""
Interactive demo: decode a few FineWeb val tokens (so you see training-domain data),
then continue text with a checkpoint trained by train_gpt.py (causal LM).
Continuation is streamed to the terminal as tokens are generated.

Usage (from repo root, after training produced final_model.pt):

  python scripts/generate_demo.py \
    --checkpoint final_model.pt \
    --tokenizer ./data/tokenizers/fineweb_1024_bpe.model

Architecture must match training (same env as train_gpt.py, e.g. VOCAB_SIZE=1024).

CPU-only is supported; CUDA/MPS used when available.

TTY: prints a small ASCII mascot + colors (like a friendly CLI assistant).
Use --plain or set NO_COLOR=1 to disable.

Without training: fetch a baseline-shaped FP16 smoke checkpoint via
  python scripts/download_cli_demo_checkpoint.py -o cli_demo_checkpoint.pt
(weights are random-init; use only to exercise the CLI / tokenizer).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train_gpt import GPT, Hyperparameters, load_data_shard  # noqa: E402


def _color_enabled() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR", "").strip() == ""


class TermStyle:
    """Tiny ANSI helper (respects NO_COLOR + TTY)."""

    def __init__(self, enabled: bool | None = None) -> None:
        self._on = _color_enabled() if enabled is None else enabled

    def wrap(self, s: str, *codes: str) -> str:
        if not self._on or not codes:
            return s
        return f"\033[{';'.join(codes)}m{s}\033[0m"

    def dim(self, s: str) -> str:
        return self.wrap(s, "2")

    def bold(self, s: str) -> str:
        return self.wrap(s, "1")

    def cyan(self, s: str) -> str:
        return self.wrap(s, "36")

    def magenta(self, s: str) -> str:
        return self.wrap(s, "35")

    def green(self, s: str) -> str:
        return self.wrap(s, "32")


def print_mascot_banner(style: TermStyle, *, plain: bool) -> None:
    """Cute terminal mascot (ASCII + optional color), Claude-Code-ish welcome."""
    if plain:
        print("Parameter Golf · generate — ready.\n")
        return
    face = style.cyan("◠ ◠")
    mouth = style.magenta("◡")
    box = style.dim("╭──────────────╮")
    box_mid = style.dim("│")
    box_bot = style.dim("╰──────┬───────╯")
    legs = style.dim("      ╱ ╲")
    title = style.bold("Parameter Golf · generate")
    print()
    print(f"      {style.dim('·   ·')}")
    print(f"    {box}")
    print(f"   {box_mid} {face} {box_mid}  {title}")
    print(f"   {box_mid}  {mouth}  {box_mid}")
    print(f"    {box_bot}")
    print(f"    {legs}")
    print()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(args: Hyperparameters) -> GPT:
    return GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    )


def show_val_sample(
    data_path: str,
    tokenizer_path: str,
    num_tokens: int,
) -> None:
    pattern = os.path.join(data_path, "fineweb_val_*.bin")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No validation shards found at {pattern}", file=sys.stderr)
        return
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    toks = load_data_shard(Path(files[0]))[:num_tokens]
    piece_ids = [int(t) for t in toks.tolist()]
    text = sp.decode(piece_ids)
    preview = text.replace("\n", " ")[:500]
    if len(text) > 500:
        preview += "..."
    print("\n--- Sample from fineweb val (first shard, decoded) ---")
    print(preview)
    print(f"({len(piece_ids)} tokens shown)\n")


def sample_next_token(
    logits_last: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
) -> int:
    """logits_last: [vocab]"""
    if temperature <= 0:
        return int(logits_last.argmax().item())
    logits = logits_last / temperature
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits = torch.where(logits < v[-1], torch.full_like(logits, float("-inf")), logits)
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


@torch.inference_mode()
def stream_continuation(
    model: GPT,
    sp: spm.SentencePieceProcessor,
    device: torch.device,
    prompt: str,
    *,
    max_new_tokens: int,
    seq_len_cap: int,
    temperature: float,
    top_k: int,
    style: TermStyle,
    plain: bool,
) -> None:
    """Print prompt then stream new tokens to stdout as they are sampled (decoded incrementally)."""
    ids = sp.encode(prompt, out_type=int)
    if not ids:
        ids = [0]
    out_ids = list(ids)
    prev_text = sp.decode(out_ids)
    if plain:
        print("\n--- Continuation ---")
    else:
        print()
        print(f"  {style.dim('╭')} {style.green('typing…')} {style.dim('╮')}")
    sys.stdout.write(prev_text)
    sys.stdout.flush()
    while len(out_ids) < len(ids) + max_new_tokens:
        ctx = out_ids[-seq_len_cap:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model.forward_logits(x)
        logits_last = logits[0, -1, :]
        next_id = sample_next_token(logits_last, temperature=temperature, top_k=top_k)
        out_ids.append(next_id)
        full = sp.decode(out_ids)
        delta = full[len(prev_text) :]
        if delta:
            sys.stdout.write(delta)
            sys.stdout.flush()
        prev_text = full
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode val sample + continuation demo for train_gpt checkpoints.")
    parser.add_argument("--checkpoint", type=str, default="final_model.pt", help="Path to final_model.pt")
    parser.add_argument("--tokenizer", type=str, default=None, help="SentencePiece .model path")
    parser.add_argument("--data-path", type=str, default=None, help="Dataset dir with fineweb_val_*.bin (for --show-sample)")
    parser.add_argument("--no-show-sample", action="store_true", help="Skip printing val shard preview")
    parser.add_argument("--sample-tokens", type=int, default=256, help="How many tokens to decode for preview")
    parser.add_argument("--prompt", type=str, default=None, help="If set, generate once and exit (non-interactive)")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50, help="0 to disable top-k")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="No colors or ASCII mascot (plain logs / CI)",
    )
    args = parser.parse_args()

    style = TermStyle(enabled=not args.plain and _color_enabled())
    hp = Hyperparameters()
    tok_path = args.tokenizer or hp.tokenizer_path
    data_path = args.data_path or hp.data_path

    if not args.no_show_sample:
        show_val_sample(data_path, tok_path, args.sample_tokens)

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(
            f"Checkpoint not found: {ckpt}\n"
            "Train with train_gpt.py first (produces final_model.pt in the cwd).",
            file=sys.stderr,
        )
        sys.exit(1)

    device = pick_device()
    sp = spm.SentencePieceProcessor(model_file=tok_path)
    model = build_model(hp).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    else:
        model = model.to(dtype=torch.float32)

    print_mascot_banner(style, plain=args.plain)
    status = f"Device: {device} | vocab_size={hp.vocab_size} | seq_len cap={hp.train_seq_len}"
    print(style.dim(status) if not args.plain else status)
    print()

    def _one(prompt: str) -> None:
        stream_continuation(
            model,
            sp,
            device,
            prompt,
            max_new_tokens=args.max_new_tokens,
            seq_len_cap=hp.train_seq_len,
            temperature=args.temperature,
            top_k=args.top_k,
            style=style,
            plain=args.plain,
        )

    if args.prompt is not None:
        _one(args.prompt)
        return

    hint = (
        "Enter prompt text (empty line to quit). Continuation streams in real time. "
        "Model is English web-text; short English works best."
    )
    print(style.dim(hint) if not args.plain else hint)
    prompt_label = "Prompt> " if args.plain else f"{style.cyan('◆')} Prompt> "
    while True:
        try:
            line = input(prompt_label).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            if not args.plain:
                print(f"  {style.magenta('· ·')} {style.dim('see you~')}")
            break
        if not line:
            break
        _one(line)


if __name__ == "__main__":
    main()
