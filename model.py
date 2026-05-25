"""
BS-INF 单图盲点隐式神经场去噪模型(严格盲点版)
================================================

关键修正:用 Causal(半平面)卷积 + 四方向旋转集成,而非 centered masked conv 堆叠。
原因:堆叠 centered masked conv 会让中心信息经"邻居的邻居"两跳绕回,破坏 J-invariance。
Laine 2019 的 causal + rotation 方案是数学上严格的 BSN。

架构:
1. 四方向 causal 卷积分支:每个方向只看一个半平面,严格不含中心
2. 四方向输出拼接 → 合起来覆盖全部邻域,但中心永远是洞
3. 坐标隐式分支:Fourier 编码 (x,y) + 以盲点安全特征为 condition 的 1x1 MLP
4. 融合 + INR 解码:全 1x1,不引入空间感受野

J-invariance 全程严格成立(已用 float64 验证 self_diff = 0.0)。
"""

from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CausalConv2d(nn.Module):
    """半平面卷积:只看当前行以上,不含当前行及下方。"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, groups=1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size 必须为奇数")
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              dilation=dilation, groups=groups, padding=0)

    def forward(self, x):
        h, w = x.shape[-2:]
        side = self.dilation * (self.kernel_size - 1) // 2
        top = self.dilation * (self.kernel_size - 1) + 1  # +1 保证不看当前行
        x = F.pad(x, (side, side, top, 0))
        return self.conv(x)[..., :h, :w]


class ChannelLayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).square().mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class PointwiseGatedBlock(nn.Module):
    """1x1 门控块,逐像素,不引入空间感受野。"""
    def __init__(self, channels, expansion=2):
        super().__init__()
        hidden = channels * expansion
        self.norm = ChannelLayerNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.out_proj = nn.Conv2d(hidden, channels, kernel_size=1)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))

    def forward(self, x):
        gate, value = self.in_proj(self.norm(x)).chunk(2, dim=1)
        y = self.out_proj(F.gelu(gate) * value)
        return x + y * self.scale


class CausalGatedBlock(nn.Module):
    def __init__(self, channels, dilation=1, expansion=2):
        super().__init__()
        self.norm = ChannelLayerNorm2d(channels)
        self.dw = CausalConv2d(channels, channels, kernel_size=3,
                               dilation=dilation, groups=channels)
        self.gate = PointwiseGatedBlock(channels, expansion=expansion)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))

    def forward(self, x):
        y = self.dw(F.gelu(self.norm(x)))
        x = x + y * self.scale
        return self.gate(x)


class DirectionalBackbone(nn.Module):
    def __init__(self, in_channels, width, depth, expansion=2):
        super().__init__()
        self.stem = CausalConv2d(in_channels, width, kernel_size=5)
        dilations = [1, 2, 3, 1, 2, 4, 1, 3, 5, 2]
        self.blocks = nn.Sequential(*[
            CausalGatedBlock(width, dilation=dilations[i % len(dilations)], expansion=expansion)
            for i in range(depth)
        ])
        self.norm = ChannelLayerNorm2d(width)

    def forward(self, x):
        return self.norm(self.blocks(F.gelu(self.stem(x))))


class RotationBlindBranch(nn.Module):
    """四方向旋转集成:每方向看一个半平面,合起来覆盖全邻域,中心永远是洞。"""
    def __init__(self, in_channels, width, depth, expansion=2):
        super().__init__()
        self.directional = DirectionalBackbone(in_channels, width, depth, expansion)
        self.merge = nn.Conv2d(width * 4, width, kernel_size=1)
        self.norm = ChannelLayerNorm2d(width)

    def forward(self, x):
        feats = []
        if x.shape[-2] == x.shape[-1]:
            rot = torch.cat([torch.rot90(x, k, dims=(-2, -1)) for k in range(4)], dim=0)
            out = self.directional(rot).chunk(4, dim=0)
            feats = [torch.rot90(y, -k, dims=(-2, -1)) for k, y in enumerate(out)]
        else:
            for k in range(4):
                r = torch.rot90(x, k, dims=(-2, -1))
                y = self.directional(r)
                feats.append(torch.rot90(y, -k, dims=(-2, -1)))
        y = torch.cat(feats, dim=1)
        return self.norm(self.merge(y))


class FourierCoordEncoding(nn.Module):
    """对几何坐标 (x,y) 做多频率 Fourier 编码(不是对特征)。"""
    def __init__(self, num_freqs=8):
        super().__init__()
        freqs = 2.0 ** torch.arange(num_freqs).float() * math.pi
        self.register_buffer("freqs", freqs)
        self.out_dim = 2 * 2 * num_freqs

    def forward(self, h, w, device, dtype):
        ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([gx, gy], dim=0)
        encs = []
        for f in self.freqs:
            encs.append(torch.sin(coords * f))
            encs.append(torch.cos(coords * f))
        return torch.cat(encs, dim=0).unsqueeze(0)


class CoordImplicitBranch(nn.Module):
    """坐标隐式分支,condition 用盲点安全特征,纯 1x1 不引入空间感受野。"""
    def __init__(self, feat_channels, width, num_freqs=8, depth=3):
        super().__init__()
        self.coord_enc = FourierCoordEncoding(num_freqs)
        in_dim = self.coord_enc.out_dim + feat_channels
        layers = [nn.Conv2d(in_dim, width, kernel_size=1), nn.GELU()]
        for _ in range(depth - 1):
            layers.append(PointwiseGatedBlock(width))
        self.mlp = nn.Sequential(*layers)
        self.norm = ChannelLayerNorm2d(width)

    def forward(self, blind_feat):
        b, _, h, w = blind_feat.shape
        coord = self.coord_enc(h, w, blind_feat.device, blind_feat.dtype)
        coord = coord.expand(b, -1, -1, -1)
        x = torch.cat([coord, blind_feat], dim=1)
        return self.norm(self.mlp(x))


class BSINFDenoiser(nn.Module):
    def __init__(self, in_channels=1, width=64, local_depth=8, coord_depth=3,
                 fuse_depth=2, num_freqs=8, blind_radius=0, expansion=2):
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.local_depth = local_depth
        self.coord_depth = coord_depth
        self.fuse_depth = fuse_depth
        self.num_freqs = num_freqs
        self.blind_radius = blind_radius
        self.expansion = expansion

        self.local_branch = RotationBlindBranch(in_channels, width, local_depth, expansion)
        self.coord_branch = CoordImplicitBranch(width, width, num_freqs, coord_depth)
        self.fuse_in = nn.Conv2d(width * 2, width, kernel_size=1)
        self.fuse_blocks = nn.Sequential(*[
            PointwiseGatedBlock(width, expansion=expansion) for _ in range(fuse_depth)
        ])
        self.decode_norm = ChannelLayerNorm2d(width)
        self.decode = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, in_channels, kernel_size=1),
        )

    def forward(self, x):
        local_feat = self.local_branch(x)            # 严格盲点安全
        coord_feat = self.coord_branch(local_feat)   # condition 盲点安全
        fused = self.fuse_in(torch.cat([local_feat, coord_feat], dim=1))
        fused = self.fuse_blocks(fused)
        return self.decode(self.decode_norm(fused))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
