# Attribution

The HI-Diff kernels in this package are derived from the official
[zhengchen1999/HI-Diff](https://github.com/zhengchen1999/HI-Diff) implementation
(NeurIPS 2023) at upstream revision
[`b3bfd167`](https://github.com/zhengchen1999/HI-Diff/tree/b3bfd167997e27f8edd57681cf70e5031a0e35f2).
The cleanup follows the inference-only port in
[porfeng/ComfyUI-PhotoDeblur](https://github.com/porfeng/ComfyUI-PhotoDeblur).

File provenance:

| This package                                           | Upstream path                                                 |
| ------------------------------------------------------ | ------------------------------------------------------------- |
| `mindor_hi_diff/hi_diff/transformer.py`                | `hi_diff/archs/Transformer_arch.py`                           |
| `mindor_hi_diff/hi_diff/latent_encoder.py`             | `hi_diff/archs/latent_encoder_arch.py`                        |
| `mindor_hi_diff/hi_diff/denoising.py`                  | `hi_diff/archs/denoising_arch.py`                             |
| `mindor_hi_diff/hi_diff/denoising_simple.py`           | `hi_diff/archs/denoising_simple_arch.py`                      |
| `mindor_hi_diff/ddpm.py`                               | `ldm/ddpm.py` (inference-only extract)                        |
| `mindor_hi_diff/ddpm_schedule.py`                      | `ldm/util2.py` (inference-only extract)                       |
| `mindor_hi_diff/gopro_schedule.py`                     | `hi_diff/utils/beta_schedule.py`                              |
| `mindor_hi_diff/gopro_prior.py`                        | `hi_diff/models/HI_Diff_S2_model.py` (inference-only extract) |

Modifications from upstream:

- Dropped BasicSR registry calls (`@ARCH_REGISTRY.register()` and the
  `from basicsr.utils.registry import ARCH_REGISTRY` imports) from every
  architecture module so the classes load standalone.
- Extracted the GoPro variant's sampling helpers (`set_new_noise_schedule`,
  `predict_start_from_noise`, `q_posterior`, `p_mean_variance`, `p_sample`,
  `p_sample_loop`) from `HI_Diff_S2_model.py` into a dedicated
  ``gopro_prior.py`` module. The rest of `HI_Diff_S2_model.py` is training
  infrastructure and is not vendored.
- Added a single public builder (`build_hi_diff_model`) that assembles the
  Restormer backbone, latent encoder and denoiser for the requested variant
  and loads the three checkpoint files.
- Flattened the directory layout into a single `mindor_hi_diff/` package so
  the kernels become importable as
  ``from mindor_hi_diff import build_hi_diff_model`` without `sys.path`
  manipulation.

Upstream license: Apache License 2.0. See [LICENSE](LICENSE).
