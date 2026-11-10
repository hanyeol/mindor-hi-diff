"""Inference-only HI-Diff (Hierarchical Integration Diffusion) deblurring.

Public entry point:
  * ``build_hi_diff_model`` — construct one of the three HI-Diff variants
    (RealBlur_J / RealBlur_R / GoPro) from already-loaded checkpoints.

The architecture classes (``Transformer``, ``DDPM``, ``GoProPrior`` …) are
intentionally not re-exported — downstream code should route every use
through ``build_hi_diff_model`` so variant differences stay encapsulated.
"""

from .model import build_hi_diff_model

__all__ = [ "build_hi_diff_model" ]
