#!/usr/bin/env python3
"""
Download (or unpack) a Parameter-Golf-shaped GPT checkpoint for testing scripts/generate_demo.py
without running train_gpt.py.

The default artifact matches train_gpt.py default Hyperparameters (9L / 512 dim / vocab 1024 /
tied embeddings). It is random initialization compressed as FP16 — useful only to verify the CLI
and streaming; generations are not meaningful.

Priority:
  1. URL from --url or PARAMETER_GOLF_DEMO_CKPT_URL
  2. Built-in raw GitHub URL (after you push demo_assets/ to that branch)
  3. Local repo file demo_assets/baseline_demo_fp16.pt.lzma

Example:
  python scripts/download_cli_demo_checkpoint.py -o cli_demo_checkpoint.pt
  python scripts/generate_demo.py --checkpoint cli_demo_checkpoint.pt --tokenizer ./data/tokenizers/fineweb_1024_bpe.model --no-show-sample
"""

from __future__ import annotations

import argparse
import io
import lzma
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
# After demo_assets/ is on your default branch, raw GitHub serves this without authentication.
_DEFAULT_RAW_GITHUB = (
    "https://raw.githubusercontent.com/openai/parameter-golf/main/demo_assets/baseline_demo_fp16.pt.lzma"
)


def _default_urls() -> list[str]:
    out: list[str] = []
    env = os.environ.get("PARAMETER_GOLF_DEMO_CKPT_URL", "").strip()
    if env:
        out.append(env)
    raw_base = os.environ.get("PARAMETER_GOLF_DEMO_RAW_BASE", "").strip().rstrip("/")
    if raw_base:
        out.append(f"{raw_base}/demo_assets/baseline_demo_fp16.pt.lzma")
    out.append(_DEFAULT_RAW_GITHUB)
    return out


def _fetch_bytes(url: str, timeout: float = 120.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "parameter-golf-cli-demo/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _load_lzma_pt(data: bytes) -> dict:
    import torch

    raw = lzma.decompress(data)
    obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise ValueError(f"checkpoint must be a state_dict dict, got {type(obj)}")
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch CLI demo GPT checkpoint (baseline-shaped).")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("cli_demo_checkpoint.pt"),
        help="Where to write uncompressed .pt (default: ./cli_demo_checkpoint.pt)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="",
        help="If set, only this URL is tried (HTTPS). Ignores local copy unless download fails.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Do not use HTTP; only unpack demo_assets/baseline_demo_fp16.pt.lzma from this repo",
    )
    parser.add_argument(
        "--prefer-remote",
        action="store_true",
        help="Try HTTP URLs before using demo_assets/ in this clone.",
    )
    args = parser.parse_args()

    import torch

    lzma_bytes: bytes | None = None
    source = ""

    local_lzma = _REPO_ROOT / "demo_assets" / "baseline_demo_fp16.pt.lzma"

    def try_remote() -> bool:
        nonlocal lzma_bytes, source
        if args.local_only:
            return False
        urls = [args.url] if args.url.strip() else _default_urls()
        for url in urls:
            try:
                print(f"Trying {url}", file=sys.stderr)
                lzma_bytes = _fetch_bytes(url)
                source = url
                return True
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                print(f"  failed: {e}", file=sys.stderr)
        return False

    if args.prefer_remote:
        try_remote()
    if lzma_bytes is None and local_lzma.is_file():
        lzma_bytes = local_lzma.read_bytes()
        source = str(local_lzma)
    if lzma_bytes is None:
        try_remote()

    if lzma_bytes is None:
        print(
            "No checkpoint bytes available. Either:\n"
            "  • Clone this repo with demo_assets/baseline_demo_fp16.pt.lzma present, or\n"
            "  • Set PARAMETER_GOLF_DEMO_CKPT_URL (or pass --url) to a direct HTTPS link to that .lzma file, or\n"
            "  • Push demo_assets/ to GitHub and use the default raw URL (see script header).\n",
            file=sys.stderr,
        )
        sys.exit(1)

    state = _load_lzma_pt(lzma_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes) from {source}")


if __name__ == "__main__":
    main()
