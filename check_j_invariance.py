"""
J-invariance 严格性验证(必做的健全性测试)
=============================================

原理:扰动输入图某个像素的值,看该像素位置的输出是否变化。
如果 J-invariance 严格成立,扰动中心像素 → 该位置输出 完全不变。

这个测试能抓出所有"中心信息泄漏"的 bug:
- 第一层用了普通卷积而非 masked conv
- 解码器引入了空间感受野
- 坐标分支泄漏了中心

用法:
    python -m bsinf_denoise.check_j_invariance       # 用随机初始化的模型测
    python -m bsinf_denoise.check_j_invariance --checkpoint runs/exp/checkpoint.pt
"""

from __future__ import annotations

import argparse

import torch

from .model import BSINFDenoiser


def check(model, device, image_size=64, num_probes=20, perturb=10.0, channels=1):
    model.eval()
    x = torch.randn(1, channels, image_size, image_size, device=device)

    with torch.no_grad():
        base = model(x)

    max_self_diff = 0.0      # 扰动点自身输出的变化(应该 ≈ 0)
    max_neighbor_diff = 0.0  # 扰动点邻居输出的变化(应该 > 0,证明网络在工作)

    torch.manual_seed(0)
    for _ in range(num_probes):
        py = torch.randint(10, image_size - 10, (1,)).item()
        px = torch.randint(10, image_size - 10, (1,)).item()

        perturbed = x.clone()
        perturbed[0, :, py, px] += perturb

        with torch.no_grad():
            new = model(perturbed)

        # 该位置自身的输出变化(J-invariance: 应为 0)
        self_diff = (new[0, :, py, px] - base[0, :, py, px]).abs().max().item()
        max_self_diff = max(max_self_diff, self_diff)

        # 邻居位置的输出变化(网络应该用中心去影响邻居预测,所以应 > 0)
        ny, nx = py + 1, px + 1
        neighbor_diff = (new[0, :, ny, nx] - base[0, :, ny, nx]).abs().max().item()
        max_neighbor_diff = max(max_neighbor_diff, neighbor_diff)

    return max_self_diff, max_neighbor_diff


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--blind-radius", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = BSINFDenoiser(**ckpt["model_cfg"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        channels = ckpt["model_cfg"]["in_channels"]
        print(f"加载 checkpoint: {args.checkpoint}")
    else:
        model = BSINFDenoiser(in_channels=1, width=32, local_depth=4,
                              coord_depth=2, fuse_depth=2,
                              blind_radius=args.blind_radius).to(device)
        channels = 1
        print("用随机初始化模型测试(验证架构本身,与训练无关)")

    self_diff, neighbor_diff = check(model, device,
                                     image_size=args.image_size, channels=channels)

    print("\n" + "=" * 60)
    print("J-invariance 验证结果")
    print("=" * 60)
    print(f"  扰动点自身输出最大变化:   {self_diff:.3e}  (应 ≈ 0)")
    print(f"  扰动点邻居输出最大变化:   {neighbor_diff:.3e}  (应 > 0)")
    print()

    ok_self = self_diff < 1e-4
    ok_neighbor = neighbor_diff > 1e-6

    if ok_self and ok_neighbor:
        print("  ✓ 通过!J-invariance 严格成立,且网络确实在用邻域信息。")
    elif not ok_self:
        print("  ✗ 失败!中心信息泄漏 —— 检查:")
        print("     - 第一层是否用了 CenteredMaskedConv2d(不能用普通 conv)")
        print("     - 解码器是否只用 1x1(不能用 3x3)")
        print("     - 坐标分支 condition 是否盲点安全")
    elif not ok_neighbor:
        print("  ⚠ 警告:邻居输出几乎不变,网络可能没在学有用的东西(或还没训练)。")
    print("=" * 60)


if __name__ == "__main__":
    main()
