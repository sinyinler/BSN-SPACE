"""
推理脚本:用训练好的 checkpoint 对 npy 去噪
=============================================
用法:
    python -m bsinf_denoise.infer --input your_bfi.npy \
        --checkpoint runs/exp/checkpoint.pt --output outputs/denoised.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .data import (
        NormMeta, load_npy, to_chw, apply_transform, invert_transform,
        save_npy, save_preview, save_residual,
    )
    from .model import BSINFDenoiser
    from .train import denoise_full, resolve_device
except ImportError:
    from data import (
        NormMeta, load_npy, to_chw, apply_transform, invert_transform,
        save_npy, save_preview, save_residual,
    )
    from model import BSINFDenoiser
    from train import denoise_full, resolve_device


def load_checkpoint(path, device):
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = ckpt["model_cfg"]
    model = BSINFDenoiser(**cfg).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    meta = NormMeta.from_dict(ckpt["meta"])
    return model, meta


def build_parser():
    p = argparse.ArgumentParser(description="BS-INF 去噪推理")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/denoised.npy"))
    p.add_argument("--device", default="auto")
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--tile-context", type=int, default=64)
    p.add_argument("--no-tta", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    model, meta = load_checkpoint(args.checkpoint, device)

    source = load_npy(args.input)
    chw, layout = to_chw(source)
    transformed = apply_transform(chw, meta)

    denoised = denoise_full(model, transformed, device,
                            tile=args.tile, context=args.tile_context,
                            tta=not args.no_tta)

    save_npy(args.output, invert_transform(denoised, meta), meta)
    out_dir = args.output.parent
    save_preview(out_dir / "denoised.png", denoised, meta)
    save_residual(out_dir / "residual.png", transformed, denoised)

    print(json.dumps(dict(
        input=str(args.input), checkpoint=str(args.checkpoint),
        output=str(args.output), device=str(device),
        tta=not args.no_tta, shape=list(transformed.shape),
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
