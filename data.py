"""
数据加载与归一化
================

支持 2D (H,W) 和 3D (C,H,W)/(H,W,C) 的 npy。
默认归一化用 percentile + 线性缩放(我们讨论过对 BFI 比 log1p 更稳),
也保留 log1p 选项供对比实验。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

Layout = Literal["HW", "CHW", "HWC"]


@dataclass
class NormMeta:
    layout: Layout
    original_shape: tuple
    transform: str          # "percentile" | "log1p" | "standard"
    low: float              # percentile 下界值(原始域)
    high: float             # percentile 上界值(原始域)
    mean: float
    std: float
    preview_low: float
    preview_high: float

    def to_dict(self):
        d = asdict(self)
        d["original_shape"] = list(self.original_shape)
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(
            layout=d["layout"],
            original_shape=tuple(int(v) for v in d["original_shape"]),
            transform=d["transform"],
            low=float(d["low"]), high=float(d["high"]),
            mean=float(d["mean"]), std=float(d["std"]),
            preview_low=float(d["preview_low"]),
            preview_high=float(d["preview_high"]),
        )


def load_npy(path) -> np.ndarray:
    arr = np.load(Path(path))
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        finite = np.isfinite(arr)
        fill = float(np.nanmedian(arr[finite])) if finite.any() else 0.0
        arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    return arr


def infer_layout(arr: np.ndarray) -> Layout:
    if arr.ndim == 2:
        return "HW"
    if arr.ndim != 3:
        raise ValueError(f"只支持 2D 或 3D npy, got shape={arr.shape}")
    if arr.shape[-1] in (1, 2, 3, 4):
        return "HWC"
    if arr.shape[0] in (1, 2, 3, 4):
        return "CHW"
    raise ValueError(f"无法判断通道布局, shape={arr.shape}")


def to_chw(arr: np.ndarray):
    layout = infer_layout(arr)
    if layout == "HW":
        chw = arr[None]
    elif layout == "HWC":
        chw = np.moveaxis(arr, -1, 0)
    else:
        chw = arr
    return np.ascontiguousarray(chw, dtype=np.float32), layout


def from_chw(chw: np.ndarray, layout: Layout):
    if layout == "HW":
        return chw[0]
    if layout == "HWC":
        return np.moveaxis(chw, 0, -1)
    return chw


def normalize(chw: np.ndarray, source: np.ndarray, layout: Layout,
              transform: str = "percentile",
              p_low: float = 0.5, p_high: float = 99.5) -> tuple:
    low, high = np.percentile(source, [p_low, p_high])
    if float(high - low) < 1e-8:
        low, high = float(source.min()), float(source.max() + 1e-6)
    mean, std = float(source.mean()), float(source.std() + 1e-8)

    meta = NormMeta(
        layout=layout, original_shape=tuple(source.shape),
        transform=transform, low=float(low), high=float(high),
        mean=mean, std=std, preview_low=float(low), preview_high=float(high),
    )
    return apply_transform(chw, meta), meta


def apply_transform(chw: np.ndarray, meta: NormMeta) -> np.ndarray:
    if meta.transform == "percentile":
        # 线性缩放到 [0,1],不裁剪(保留极值的相对关系)
        return ((chw - meta.low) / (meta.high - meta.low)).astype(np.float32)
    if meta.transform == "log1p":
        if float(chw.min()) <= -1.0:
            raise ValueError("log1p 需要值 > -1")
        return np.log1p(chw).astype(np.float32)
    if meta.transform == "standard":
        return ((chw - meta.mean) / meta.std).astype(np.float32)
    raise ValueError(f"未知 transform: {meta.transform}")


def invert_transform(chw: np.ndarray, meta: NormMeta) -> np.ndarray:
    if meta.transform == "percentile":
        return (chw * (meta.high - meta.low) + meta.low).astype(np.float32)
    if meta.transform == "log1p":
        return np.expm1(chw).astype(np.float32)
    if meta.transform == "standard":
        return (chw * meta.std + meta.mean).astype(np.float32)
    raise ValueError(f"未知 transform: {meta.transform}")


def save_npy(path, chw: np.ndarray, meta: NormMeta):
    out = from_chw(chw, meta.layout)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(Path(path), out.astype(np.float32))


def _preview(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    scaled = (arr - low) / max(high - low, 1e-8)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def save_preview(path, chw_norm: np.ndarray, meta: NormMeta):
    """chw_norm 是归一化域的数据,先还原再存 png"""
    arr = from_chw(invert_transform(chw_norm, meta), meta.layout)
    png = _preview(arr, meta.preview_low, meta.preview_high)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if png.ndim == 2:
        Image.fromarray(png, "L").save(path)
    elif png.ndim == 3 and png.shape[-1] in (3, 4):
        Image.fromarray(png).save(path)
    else:
        Image.fromarray(png[..., 0] if png.ndim == 3 else png, "L").save(path)


def save_residual(path, noisy_norm: np.ndarray, denoised_norm: np.ndarray):
    res = np.abs(noisy_norm - denoised_norm)
    high = float(np.percentile(res, 99.0))
    png = _preview(res, 0.0, high if high > 1e-8 else 1.0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = png[0] if png.ndim == 3 else png
    Image.fromarray(arr, "L").save(path)
