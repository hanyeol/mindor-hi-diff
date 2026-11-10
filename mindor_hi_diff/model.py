from typing import Any, Dict, Literal, Mapping
import torch
from torch import Tensor
from .hi_diff.latent_encoder import latent_encoder_gelu, latent_encoder_lrelu
from .hi_diff.transformer import Transformer

HiDiffVariant = Literal[ "RealBlur_J", "RealBlur_R", "GoPro" ]

_HI_DIFF_VARIANTS = ( "RealBlur_J", "RealBlur_R", "GoPro" )

class HiDiffModel(torch.nn.Module):
    def __init__(self, net_g: torch.nn.Module, prior: torch.nn.Module) -> None:
        super().__init__()

        self.net_g = net_g
        self.prior = prior

    def forward(self, image: Tensor) -> Tensor:
        return self.net_g(image, self.prior(image))

def build_hi_diff_model(variant: HiDiffVariant, checkpoints: Dict[str, Mapping[str, Any]]) -> torch.nn.Module:
    """Build a HI-Diff inference model for ``variant`` and load ``checkpoints``.

    ``checkpoints`` holds the three state dicts the model needs, keyed
    ``"g"`` (Restormer backbone), ``"le_dm"`` (latent encoder) and ``"d"``
    (denoiser). Each value may be the raw state dict or the full checkpoint
    dict (with a ``params`` key). ``module.`` prefixes left over from
    DataParallel training are stripped automatically.

    Returns an eval-mode ``torch.nn.Module`` on the CPU with gradients
    disabled; move it to the target device after construction.
    """
    if variant not in _HI_DIFF_VARIANTS:
        raise ValueError(f"Unsupported HI-Diff variant: {variant!r}; expected one of {_HI_DIFF_VARIANTS}")

    is_gopro = variant == "GoPro"
    steps    = 8 if is_gopro else 16

    net_g = Transformer(
        inp_channels=3, out_channels=3,
        dim=48, num_blocks=[ 3, 5, 6, 6 ],
        num_refinement_blocks=4, heads=[ 1, 2, 4, 8 ],
        ffn_expansion_factor=2.66 if is_gopro else 2,
        bias=False, LayerNorm_type="WithBias", dual_pixel_task=False,
        embed_dim=64, group=4,
    )

    encoder = latent_encoder_gelu if is_gopro else latent_encoder_lrelu
    condition = encoder(
        in_chans=3, embed_dim=64,
        block_num=6 if is_gopro else 5,
        group=4, stage=2, patch_expansion=0.5, channel_expansion=4,
    )

    if is_gopro:
        from .hi_diff.denoising import denoising
        from .gopro_prior import GoProPrior

        denoise = denoising(
            in_channel=256, out_channel=256, inner_channel=512,
            block_num=4, group=4, patch_expansion=0.5, channel_expansion=2,
        )
    else:
        from .hi_diff.denoising_simple import simple_denoise
        from .ddpm import DDPM

        denoise = simple_denoise(n_feats=64, n_denoise_res=4)

    net_g.load_state_dict(_strip_module_prefix(checkpoints["g"], "params"), strict=True)
    condition.load_state_dict(_strip_module_prefix(checkpoints["le_dm"], "params"), strict=True)
    denoise.load_state_dict(_strip_module_prefix(checkpoints["d"], "params"), strict=True)

    if is_gopro:
        prior = GoProPrior(condition, denoise, steps)
    else:
        prior = DDPM(
            denoise=denoise, condition=condition, n_feats=64, group=4,
            linear_start=0.1, linear_end=0.99, timesteps=steps,
        )

    return HiDiffModel(net_g, prior).requires_grad_(False).eval()

def _strip_module_prefix(checkpoint: Mapping[str, Any], key: str) -> Dict[str, Any]:
    # HI-Diff wraps weights under 'params'; some are already flat. DataParallel
    # training also prepends 'module.' to every key — drop it so strict=True holds.
    state_dict: Dict[str, Any] = {}

    for name, value in checkpoint.get(key, checkpoint).items():
        state_dict[name.removeprefix("module.")] = value

    return state_dict
