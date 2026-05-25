"""
训练脚本:单图 BS-INF 自监督去噪
=================================

基于我们讨论的所有结论的默认配置:
- 损失:MSE(在 BSN 框架下比 Charbonnier 好,因为学到的是邻域均值=真值)
- 归一化:percentile(对 BFI 比 log1p 稳)
- blind_radius:0(你的噪声相关核 <= 1 像素,最小盲点即可)
- EMA + cosine annealing + grad clip + TTA

用法:
    python -m bsinf_denoise.train --input your_bfi.npy --out runs/exp1 --preset balanced

针对 24G A500 的预设已调好,不会 OOM。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from tqdm import trange
except ImportError:
    def trange(n, **kw):
        return range(n)

try:
    from .data import (
        NormMeta, load_npy, to_chw, normalize, invert_transform,
        save_npy, save_preview, save_residual, from_chw,
    )
    from .model import BSINFDenoiser, count_parameters
except ImportError:
    from data import (
        NormMeta, load_npy, to_chw, normalize, invert_transform,
        save_npy, save_preview, save_residual, from_chw,
    )
    from model import BSINFDenoiser, count_parameters


# ============================================================
# 预设(针对 24G A500 调好)
# ============================================================
PRESETS = {
    "fast": dict(
        steps=3000, width=32, local_depth=5, coord_depth=2, fuse_depth=2,
        num_freqs=6, patch_size=128, batch_size=4, lr=1e-3,
    ),
    "balanced": dict(
        steps=8000, width=64, local_depth=8, coord_depth=3, fuse_depth=2,
        num_freqs=8, patch_size=192, batch_size=4, lr=8e-4,
    ),
    "best": dict(
        steps=15000, width=96, local_depth=10, coord_depth=4, fuse_depth=3,
        num_freqs=10, patch_size=256, batch_size=2, lr=6e-4,
    ),
}


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# Patch 采样器
# ============================================================
class PatchSampler:
    def __init__(self, image: np.ndarray, patch_size: int, batch_size: int):
        self.image = torch.from_numpy(image).float()
        self.patch_size = patch_size
        self.batch_size = batch_size
        _, h, w = self.image.shape
        # 图比 patch 小则 pad
        pad_h = max(0, patch_size - h)
        pad_w = max(0, patch_size - w)
        if pad_h or pad_w:
            l, r = pad_w // 2, pad_w - pad_w // 2
            t, b = pad_h // 2, pad_h - pad_h // 2
            self.image = F.pad(self.image[None], (l, r, t, b), mode="reflect").squeeze(0)

    def sample(self) -> Tensor:
        _, h, w = self.image.shape
        patches = []
        for _ in range(self.batch_size):
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            p = self.image[:, top:top + self.patch_size, left:left + self.patch_size]
            # 数据增强:随机旋转 + 翻转
            turns = random.randint(0, 3)
            p = torch.rot90(p, turns, dims=(-2, -1))
            if random.random() < 0.5:
                p = torch.flip(p, dims=(-1,))
            if random.random() < 0.5:
                p = torch.flip(p, dims=(-2,))
            patches.append(p.contiguous())
        return torch.stack(patches, dim=0)


# ============================================================
# 损失:MSE(默认) / Charbonnier(对比用)
# ============================================================
def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(pred, target)


def charbonnier_loss(pred: Tensor, target: Tensor, eps: float = 1e-3) -> Tensor:
    return torch.sqrt((pred - target).square() + eps * eps).mean()


def crop_for_loss(pred: Tensor, target: Tensor, margin: int):
    """裁掉边界,避免 padding 影响损失"""
    if margin <= 0 or pred.shape[-1] <= 2 * margin:
        return pred, target
    return pred[..., margin:-margin, margin:-margin], target[..., margin:-margin, margin:-margin]


# ============================================================
# EMA
# ============================================================
def update_ema(model: nn.Module, ema: nn.Module, decay: float):
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema.state_dict().items():
            mv = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(mv.detach(), alpha=1 - decay)
            else:
                v.copy_(mv)


# ============================================================
# TTA 推理(8 方向几何平均)
# ============================================================
@torch.no_grad()
def denoise_tta(model: nn.Module, x: Tensor, tta: bool = True) -> Tensor:
    if not tta:
        return model(x)
    outs = []
    for flip in (False, True):
        base = torch.flip(x, dims=(-1,)) if flip else x
        for turns in range(4):
            aug = torch.rot90(base, turns, dims=(-2, -1))
            pred = model(aug)
            pred = torch.rot90(pred, -turns, dims=(-2, -1))
            if flip:
                pred = torch.flip(pred, dims=(-1,))
            outs.append(pred)
    return torch.stack(outs, dim=0).mean(dim=0)


# ============================================================
# 分块推理(大图防 OOM)
# ============================================================
@torch.no_grad()
def denoise_full(model: nn.Module, image: np.ndarray, device,
                 tile: int = 512, context: int = 64, tta: bool = True) -> np.ndarray:
    c, h, w = image.shape
    x = torch.from_numpy(image[None]).float().to(device)
    if h <= tile and w <= tile:
        return denoise_tta(model, x, tta=tta).squeeze(0).cpu().numpy()

    # 滑窗 + halo context + cosine blending
    out = torch.zeros_like(x)
    weight = torch.zeros(1, 1, h, w, device=device)
    step = tile - 2 * context
    ys = list(range(0, max(1, h - context), step))
    xs = list(range(0, max(1, w - context), step))
    for ty in ys:
        for tx in xs:
            y0 = max(0, ty - context); y1 = min(h, ty + tile - context)
            x0 = max(0, tx - context); x1 = min(w, tx + tile - context)
            patch = x[:, :, y0:y1, x0:x1]
            pred = denoise_tta(model, patch, tta=tta)
            # cosine ramp 权重
            ph, pw = pred.shape[-2:]
            wy = torch.hann_window(ph, device=device).clamp_min(0.05)
            wx = torch.hann_window(pw, device=device).clamp_min(0.05)
            wmask = (wy[:, None] * wx[None, :])[None, None]
            out[:, :, y0:y1, x0:x1] += pred * wmask
            weight[:, :, y0:y1, x0:x1] += wmask
    out = out / weight.clamp_min(1e-8)
    return out.squeeze(0).cpu().numpy()


# ============================================================
# 主训练
# ============================================================
def train(args):
    if args.preset:
        for k, v in PRESETS[args.preset].items():
            if getattr(args, k, None) is None or args._from_preset:
                setattr(args, k, v)
    seed_all(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # 数据
    source = load_npy(args.input)
    chw, layout = to_chw(source)
    transformed, meta = normalize(chw, source, layout,
                                  transform=args.transform,
                                  p_low=args.p_low, p_high=args.p_high)
    sampler = PatchSampler(transformed, args.patch_size, args.batch_size)

    # 模型
    model = BSINFDenoiser(
        in_channels=transformed.shape[0],
        width=args.width, local_depth=args.local_depth,
        coord_depth=args.coord_depth, fuse_depth=args.fuse_depth,
        num_freqs=args.num_freqs, blind_radius=args.blind_radius,
    ).to(device)
    ema = copy.deepcopy(model).eval()
    print(f"模型参数量: {count_parameters(model)/1e6:.2f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loss_fn = mse_loss if args.loss == "mse" else (lambda p, t: charbonnier_loss(p, t, args.charbonnier_eps))

    args.out.mkdir(parents=True, exist_ok=True)
    save_preview(args.out / "noisy.png", transformed, meta)

    # warmup
    warmup_steps = min(500, args.steps // 10)

    running = math.nan
    pbar = trange(args.steps, desc="training", dynamic_ncols=True)
    for step in pbar:
        batch = sampler.sample().to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        # lr warmup
        if step < warmup_steps:
            for g in opt.param_groups:
                g["lr"] = args.lr * (step + 1) / warmup_steps
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            pred = model(batch)
            pl, tl = crop_for_loss(pred, batch, args.loss_crop)
            loss = loss_fn(pl.float(), tl.float())
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        old_scale = scaler.get_scale()
        scaler.step(opt); scaler.update()
        if step >= warmup_steps and (not use_amp or scaler.get_scale() >= old_scale):
            sched.step()
        update_ema(model, ema, args.ema_decay)

        v = float(loss.detach().cpu())
        running = v if math.isnan(running) else 0.98 * running + 0.02 * v
        if hasattr(pbar, "set_postfix") and (step % args.log_every == 0):
            pbar.set_postfix(loss=f"{running:.5f}", lr=f"{opt.param_groups[0]['lr']:.2e}")

        # 定期存中间 checkpoint(方便挑最佳)
        if args.save_interval > 0 and (step + 1) % args.save_interval == 0:
            _save_ckpt(ema, meta, args, args.out / f"ckpt_{step+1}.pt")

    # 最终推理
    ema.eval()
    denoised = denoise_full(ema, transformed, device,
                            tile=args.tile, context=args.tile_context, tta=not args.no_tta)
    save_npy(args.out / "denoised.npy", invert_transform(denoised, meta), meta)
    save_preview(args.out / "denoised.png", denoised, meta)
    save_residual(args.out / "residual.png", transformed, denoised)
    _save_ckpt(ema, meta, args, args.out / "checkpoint.pt")

    residual = transformed - denoised
    metrics = dict(
        input=str(args.input), shape=list(transformed.shape),
        device=str(device), preset=args.preset, steps=args.steps,
        loss=args.loss, transform=args.transform, blind_radius=args.blind_radius,
        width=args.width, parameters=count_parameters(model),
        final_loss_ema=running, residual_std=float(residual.std()),
    )
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def _save_ckpt(model, meta: NormMeta, args, path):
    torch.save(dict(
        model_cfg=dict(
            in_channels=int(meta.original_shape[0]) if meta.layout == "CHW" else 1,
            width=args.width, local_depth=args.local_depth,
            coord_depth=args.coord_depth, fuse_depth=args.fuse_depth,
            num_freqs=args.num_freqs, blind_radius=args.blind_radius,
        ),
        state_dict=model.state_dict(),
        meta=meta.to_dict(),
    ), path)


def build_parser():
    p = argparse.ArgumentParser(description="单图 BS-INF 自监督去噪训练")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs/exp"))
    p.add_argument("--preset", choices=sorted(PRESETS), default="balanced")
    # 模型(不传则用 preset)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--local-depth", type=int, default=None)
    p.add_argument("--coord-depth", type=int, default=None)
    p.add_argument("--fuse-depth", type=int, default=None)
    p.add_argument("--num-freqs", type=int, default=None)
    p.add_argument("--blind-radius", type=int, default=0)
    # 训练
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--patch-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--loss-crop", type=int, default=8)
    # 损失 / 归一化
    p.add_argument("--loss", choices=["mse", "charbonnier"], default="mse")
    p.add_argument("--charbonnier-eps", type=float, default=1e-3)
    p.add_argument("--transform", choices=["percentile", "log1p", "standard"], default="percentile")
    p.add_argument("--p-low", type=float, default=0.5)
    p.add_argument("--p-high", type=float, default=99.5)
    # 推理
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--tile-context", type=int, default=64)
    p.add_argument("--no-tta", action="store_true")
    # 杂项
    p.add_argument("--amp", action="store_true", help="混合精度(A500 建议开)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=0, help="每 N 步存 checkpoint, 0 关闭")
    return p


def main():
    args = build_parser().parse_args()
    # 标记哪些参数没手动传(用 preset 填充)
    args._from_preset = True
    for k in ["width", "local_depth", "coord_depth", "fuse_depth",
              "num_freqs", "steps", "patch_size", "batch_size", "lr"]:
        if getattr(args, k) is not None:
            args._from_preset = False
    # 简单处理:逐个用 preset 填 None
    preset = PRESETS[args.preset]
    for k, v in preset.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    train(args)


if __name__ == "__main__":
    main()
