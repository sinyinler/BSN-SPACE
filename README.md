# BS-INF 单图盲点隐式神经场去噪

针对**单张 npy 图片**的自监督去噪项目,专为 BFI(血流成像)这类**噪声逐像素独立、信号有强空间冗余**的数据设计。整合了我们讨论中的所有结论。

## 设计要点(为什么这样设计)

1. **严格盲点(J-invariance)**:用 Causal(半平面)卷积 + 四方向旋转集成,而**不是** centered masked conv 堆叠。
   后者会让中心信息经"邻居的邻居"两跳绕回,破坏盲点性。本项目已用 float64 验证 `self_diff = 0.000`。
2. **BS-INF 两路结构**:局部盲点分支(扩大感受野)+ 坐标隐式分支(Fourier 编码 (x,y) 几何坐标,缓解噪声残留 / 大血管不光滑)。
3. **MSE 损失(默认)**:在 BSN 框架下 MSE 学到的是邻域**均值**=真值(噪声零均值时),
   而 Charbonnier 学到的是**中位数**,对非对称的 BFI 噪声会有系统性偏置 → 伪影。这与监督学习的经验相反。
4. **percentile 归一化(默认)**:对 BFI 比 log1p 更稳(log1p 会反转异方差噪声的优先级,且让正则项相对权重失衡)。
5. **blind_radius=0**:你的噪声相关核 ≤ 1 像素(差分图 FWHM=0),用最小盲点即可,无需大盲区。

## 安装

```bash
pip install torch numpy pillow tqdm
```

## 数据格式

单个 `.npy` 文件,支持:
- 2D 灰度图 `(H, W)`  ← BFI 最常见
- 3D `(C, H, W)` 或 `(H, W, C)`,C ∈ {1,2,3,4}

## 用法

### 1. 先验证 J-invariance(建议每次改模型后都跑)

```bash
python -m bsinf_denoise.check_j_invariance
# 期望: self_diff ≈ 0, neighbor_diff > 0
```

### 2. 训练(24G A500 预设已调好,开 --amp)

```bash
# 推荐从 balanced 开始
python -m bsinf_denoise.train --input your_bfi.npy --out runs/exp1 --preset balanced --amp

# 快速试跑
python -m bsinf_denoise.train --input your_bfi.npy --out runs/fast --preset fast --amp

# 最高质量(更大网络 + 更多步)
python -m bsinf_denoise.train --input your_bfi.npy --out runs/best --preset best --amp
```

输出在 `runs/exp1/`:`denoised.npy`(去噪结果)、`denoised.png`/`noisy.png`/`residual.png`(预览)、`checkpoint.pt`、`metrics.json`。

### 3. 用 checkpoint 推理新图

```bash
python -m bsinf_denoise.infer --input another.npy --checkpoint runs/exp1/checkpoint.pt --output out/denoised.npy
```

## 预设(针对 24G A500)

| 预设 | steps | width | patch | 显存占用 | 适用 |
|------|-------|-------|-------|---------|------|
| fast | 3000 | 32 | 128 | 低 | 快速验证 |
| balanced | 8000 | 64 | 192 | 中 | 日常使用(推荐) |
| best | 15000 | 96 | 256 | 较高 | 追求质量 |

显存不够就调小 `--patch-size` 或 `--batch-size`。

## 调参建议(基于我们讨论)

- **默认就用 MSE + percentile**,不要轻易换 Charbonnier/log1p(在 BSN 框架下通常更差)。
- 想做对比实验:`--loss charbonnier`、`--transform log1p` 单独切换,一次只改一个变量。
- **小血管被磨平**是 BSN 范式的固有局限(看不到中心 → 恢复不了亚像素孤立细节),
  BS-INF 能改善噪声残留和大血管平滑度,但救不了小血管。若小血管是核心诉求,考虑 R2R 或回到 N2N。
- 想看是否过拟合:`--save-interval 1000` 存中间 checkpoint,分别推理挑视觉最好的。

## 预期效果(管理预期)

- 明显优于无去噪;接近你的 N2N baseline(单图 < 双图是信息论硬约束,很难超越)。
- 背景明显更干净,大血管对比度提升;小血管可能仍有磨平。

## 文件结构

```
bsinf_denoise/
├── model.py              # BS-INF 模型(causal + rotation 严格盲点)
├── data.py               # npy 加载 / 归一化 / 预览
├── train.py              # 训练(EMA + cosine + warmup + grad clip + TTA + 分块推理)
├── infer.py              # 独立推理
├── check_j_invariance.py # J-invariance 验证(必做)
└── __init__.py
```
