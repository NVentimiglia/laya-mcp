"""Fine-tune public Laya on gitignored SIE train JSON.

The pip `laya` package (0.1.6) has no `agent.finetune()`. This trains
the loaded DecisionModel with the same proper scoring rule the
checkpoint used, then overwrites `model.safetensors` in a copy of the
public snapshot.

  python laya-tuned/convert_sie.py
  python laya-tuned/train.py
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
DEFAULT_BASE = "convaiinnovations/laya"
DEFAULT_OUT = HERE / "checkpoints" / "laya-sie"
DEFAULT_TRAIN = HERE / "train_sie.json"


def _copy_base(src: str, dest: Path) -> None:
    if not Path(src).exists():
        src = snapshot_download(src)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        src,
        dest,
        ignore=shutil.ignore_patterns(
            ".gitattributes", "eval", "README.md", "*.py", ".git"
        ),
    )


def _encode_row(agent: Any, row: dict[str, Any]) -> dict[str, Any]:
    from laya.common import QTYPES, build_sequence, render_options

    q = agent._to_internal(row["questions"]["answer"])
    max_len = agent.cfg.get("max_len", 512)
    head_max = agent.cfg.get("head_max_len", 192)
    seq, markers = build_sequence(agent.tok, row["state"], q, max_len, head_max)
    letter = row["answers"]["answer"]["choice"]
    keys = list(q["crit"].keys()) if isinstance(q.get("crit"), dict) else []
    if letter not in keys:
        raise ValueError(f"answer {letter!r} not in {keys}")
    if len(markers) != len(render_options(q)):
        raise ValueError("option/marker mismatch")
    target = [0.0] * len(markers)
    target[keys.index(letter)] = 1.0
    return {
        "ids": seq,
        "markers": markers,
        "qtype": QTYPES[q["t"]],
        "label": keys.index(letter),
        "target": target,
    }


def _maybe_freeze_encoder(model: Any, freeze: bool) -> None:
    if not freeze:
        return
    for param in model.encoder.parameters():
        param.requires_grad = False
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Encoder frozen. Trainable params: {n:,}", flush=True)


def train(
    base: str,
    train_path: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    freeze_encoder: bool,
    device: str | None,
    copy_base: bool,
) -> None:
    import laya
    from laya.common import collate_items, proper_reward

    rows: list[dict[str, Any]] = json.loads(
        train_path.read_text(encoding="utf-8")
    )
    if len(rows) < 50:
        raise SystemExit(f"Need more train rows in {train_path}, got {len(rows)}")

    if copy_base:
        print(f"Copying {base} -> {out_dir}", flush=True)
        _copy_base(base, out_dir)

    print(f"Loading {out_dir} for training ({len(rows)} rows)…", flush=True)
    kwargs: dict[str, Any] = {}
    if device:
        kwargs["device"] = device
    agent = laya.load(str(out_dir), **kwargs)
    print(f"Device {agent.device}  dtype {agent.dtype}", flush=True)

    items = [_encode_row(agent, row) for row in rows]
    model = agent.model
    model.float()
    _maybe_freeze_encoder(model, freeze_encoder)
    model.train()
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
    )
    pad = agent.tok.pad_token_id
    device_t = agent.device
    t0 = time.perf_counter()
    steps = 0
    for epoch in range(epochs):
        perm = torch.randperm(len(items)).tolist()
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(perm), batch_size):
            batch_items = [items[i] for i in perm[start:start + batch_size]]
            b = collate_items([batch_items], pad)
            if b is None or "target" not in b:
                continue
            opt.zero_grad(set_to_none=True)
            logits, _act = model(
                b["input_ids"].to(device_t),
                b["attention_mask"].to(device_t),
                b["marker_pos"].to(device_t),
                b["marker_mask"].to(device_t),
                b["qtype"].to(device_t),
            )
            q = torch.softmax(logits.float(), dim=-1)
            target = b["target"].to(device_t)
            mask = b["marker_mask"].to(device_t).float()
            reward = proper_reward(q, target, b["qtype"].to(device_t), mask)
            loss = -reward.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0
            )
            opt.step()
            total_loss += float(loss.detach())
            n_batches += 1
            steps += 1
            if steps % 20 == 0:
                print(f"  step {steps}  loss {float(loss.detach()):.4f}", flush=True)
        avg = total_loss / max(n_batches, 1)
        elapsed = time.perf_counter() - t0
        print(
            f"epoch {epoch + 1}/{epochs}  loss {avg:.4f}  "
            f"steps {steps}  {elapsed:.0f}s",
            flush=True,
        )

    model.eval()
    cpu_sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    weights = out_dir / "model.safetensors"
    save_file(cpu_sd, str(weights))
    print(f"Saved {weights}", flush=True)
    del agent, model, opt, items
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cuda_broken() -> bool:
    if not torch.cuda.is_available():
        return True
    try:
        t = torch.tensor([2.0], device="cuda")
        _ = t * t
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"CUDA probe failed: {exc}", flush=True)
        return True


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Fine-tune Laya on SIE train JSON")
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args(argv)
    if not args.train.is_file():
        raise SystemExit(
            f"Missing {args.train}. Run: python laya-tuned/convert_sie.py"
        )

    use_cpu = args.cpu or _cuda_broken()
    freeze = args.freeze_encoder or use_cpu
    batch = 1 if use_cpu else args.batch_size
    device = "cpu" if use_cpu else "cuda"
    print(
        f"Plan: device={device} freeze_encoder={freeze} "
        f"batch={batch} epochs={args.epochs}",
        flush=True,
    )
    try:
        train(
            args.base, args.train, args.out, args.epochs, batch, args.lr,
            freeze, device, copy_base=True,
        )
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        if device == "cpu":
            raise
        print(f"GPU train failed ({exc}); retrying CPU, freeze encoder", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        train(
            args.base, args.train, args.out, args.epochs, 1, args.lr,
            True, "cpu", copy_base=False,
        )


if __name__ == "__main__":
    main()
