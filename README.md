# mindor-hi-diff

Inference-only [HI-Diff](https://github.com/zhengchen1999/HI-Diff) (Hierarchical Integration Diffusion) deblurring kernels, packaged as a minimal `pip`-installable module.

The upstream repository ships as a research codebase: it depends on the `basicsr` registry for architecture discovery, pulls in training-only entry points, and assumes a specific directory layout. This package extracts only the inference path so downstream projects can run HI-Diff with just `torch` and `einops`.

## Install

```bash
pip install git+https://github.com/hanyeol/mindor-hi-diff.git@v0.1.0
```

## Usage

```python
import torch
from mindor_hi_diff import build_hi_diff_model

# Load the three checkpoint files for the chosen variant.
# Each variant archive on the official Google Drive contains:
#   net_g.pth, net_le_dm.pth, net_d.pth
checkpoints = {
    "g":     torch.load("net_g.pth",     map_location="cpu"),
    "le_dm": torch.load("net_le_dm.pth", map_location="cpu"),
    "d":     torch.load("net_d.pth",     map_location="cpu"),
}

model = build_hi_diff_model("RealBlur_J", checkpoints).to("cuda")

# image: (1, 3, H, W) float32 in [0, 1], H and W multiples of 8.
output = model(image)  # same shape, deblurred
```

## Variants

| Variant      | Training data                  | Diffusion steps |
| ------------ | ------------------------------ | --------------- |
| `RealBlur_J` | Real-world JPEG shots          | 16              |
| `RealBlur_R` | Real-world RAW checkpoint      | 16              |
| `GoPro`      | Synthetic GoPro motion blur    | 8               |

The GoPro variant uses a larger prior network and separate sampling helper (`GoProPrior`); the two RealBlur variants share a simpler DDPM path. ``build_hi_diff_model`` picks the right assembly automatically from the ``variant`` argument.

## Compatibility

- **PyTorch**: `>=2.0`.
- **Input**: ``(B, 3, H, W)`` float32 tensors with H and W divisible by 8.
- **Precision**: fp32 only — the Restormer backbone produces NaNs in fp16.
- **GPU**: any CUDA / MPS / CPU device torch supports. Pure-PyTorch forward; no custom CUDA kernels.

See [NOTICE.md](NOTICE.md) for upstream attribution and the full list of modifications.
