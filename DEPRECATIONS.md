# Deprecations & Triage

This file records the disposition of every method/module that no longer works out of
the box after maua was resurrected against a modern environment (Python 3.12, torch
2.10, current `huggingface_hub`/`transformers`/`diffusers`).

Status legend:
- **fixed** — repaired in place, covered by tests.
- **needs-dep** — code is fine, but an optional third-party dependency is missing or
  can't be installed on a modern stack. Left importable-on-demand; xfailed in the
  fast import test with the dep named.
- **needs-user** — blocked on an action only the repo owner can take (e.g. re-adding a
  git submodule). Documented below.
- **quarantine** — genuinely broken against the modern stack and not fixable within the
  triage budget (~1h); slated to move to `maua/legacy/` during the Phase C restructure.
  Nothing is ever deleted.

The living machine-readable version of this board is `XFAIL_IMPORTS` in
`tests/fast/test_imports.py`.

---

## Weight hosting

All three original weight hosts are dead:

| Host | Status |
|---|---|
| `bearsharktopus.b-cdn.net` | 504 |
| `the-eye.eu` | expired TLS cert |
| `share.marqt40.com` | DNS gone |
| `dall-3.com` (glid3xl) | dead |

Weights are re-hosted on the Hugging Face Hub at **`wav/maua-weights`** and fetched
through the single chokepoint `maua/ops/download.py`. Resolution order:
`$MAUA_MODELZOO` (default `./modelzoo`) → HF Hub (`wav/maua-weights` or a named repo) →
original URL fallback.

Re-hosted so far: `secondary_model_imagenet_2.pth`, `glid3xl-kl-f8.pt`,
`256x256_diffusion_uncond.pt`, `512x512_diffusion_uncond_finetune_008100.pt`,
`glid3xl-bert.pt`, `glid3xl-finetune.pt`, `RIFE_HDv2.3/`.

Stable-Diffusion 1.x and pinkney weights resolve from their upstream HF repos
(`CompVis/*`, `lambdalabs/*`) directly. RIFE 2.0–2.2 / 2.4 weights were not present in
the local model zoo, so only 2.3 is re-hosted; the other versions still rely on their
original gdrive IDs.

---

## Fixed

| Method / module | What broke | Fix |
|---|---|---|
| `maua/flow/consistency.py` | `np.int` removed | `int` |
| perceptor VGG/NIMA, `nca/train.py` | torchvision `pretrained=True` removed | `weights=` enums |
| `diffusion/processors/guided.py`, `stable.py`, `glid3xl.py` | dead weight URLs | routed through `maua/ops/download.py` → HF |
| `super/video/framerate/rife.py` | dead weight URLs | `fetch_folder` → HF for v2.x |
| `diffusion/processors/glid3xl.py` | torch 2.6+ `weights_only=True` default broke fully-pickled `AutoencoderKL` | `weights_only=False` at the trusted call site |
| `nca/train.py`, `nca/generate.py` | script-style module-level `sys.argv`/model loads; removed `maua_utils`/`NCA_train` | wrapped in `train()`/`generate()` with `__main__` guards; `train()` now returns a final checkpoint path |
| `diffusion/interpolate.py` | script-style; real bug (slerp branch wrote `out`, decode read `interpolated`) | wrapped in `interpolate_images(...)`, bug fixed |
| `diffusion/interp_loop.py` | script-style, unfinished experiment | wrapped in `main()` + guard, documented |
| `GAN/projector.py` | hardcoded `/home/hans` paths, script-style | `project(model_file, file, out_dir, ..., steps)` |
| `GAN/icgan/generate.py` | `sys.path` off-by-one to submodules | fixed path |
| `audiovisual/patches/**`, `audiovisual/patches/primitives/latents.py` | stale relative-ish imports after renames | absolute imports |
| torchvision compat | `torchvision.models.utils` & `transforms.functional_tensor` removed (needed by basicsr/ic_gan/submodules) | sys.modules shims in `maua/ops/compat.py`, installed from `maua/__init__.py` |
| dep pins | `pkg_resources` & PL 1.x APIs | `setuptools<81`, `pytorch-lightning<2` in `setup.py` |
| `workspace/examples/audio/*.mp3` | committed files were truncated/corrupt | replaced with clean full-length stream copies |

## needs-dep (optional third-party deps unavailable on a modern stack)

| Module | Missing dep |
|---|---|
| `GAN/training/trainer`, `GAN/training/dataset/image` | `ffcv` (unmaintained, compile-heavy) |
| `GAN/training/train_v0` | `padl` (abandoned) |
| `GAN/training/models/experimental/deepinvolutional` | `involution` |
| `GAN/training/models/experimental/equivariant` | `escnn` (unsatisfiable pins) |
| `GAN/training/models/experimental/stylehypermixerfly` | `torch_butterfly` |
| `audiovisual/audioreactive/selfsupervised/features/correlation` | `anatome` (unsatisfiable pins) |
| `audiovisual/audioreactive/selfsupervised/features/efficient_quantile` | C++ extension; build in place via its `setup.py` |

## needs-user

| Module | Blocker |
|---|---|
| `GAN/ZSSGAN`, `GAN/nada` | Import `maua.GAN.pix2pix`, a git submodule whose gitlink was never committed (it's in `.gitmodules` but not the index). Re-add with:<br>`git submodule add https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix maua/GAN/pix2pix`<br>The automated permission classifier blocks me from adding an external repo you didn't name, so this one is yours to run. |

## quarantine (Phase C → `maua/legacy/`)

| Module | Why |
|---|---|
| `autoregressive/min_dalle/generate` | minDALL-E submodule uses mutable dataclass defaults, rejected by Python ≥3.12 (would need forking upstream) |
| `autoregressive/rq_dalle` | rqvae submodule, same mutable-dataclass-default problem |
| `autoregressive/ru_dalle` (`api`, `generate`, `finetune`) | `rudalle` needs `huggingface_hub.cached_download`, removed from modern `huggingface_hub` |
| `autoregressive/cog/video/{generate,infinite}` | CogVideo / `icetk` need protobuf<3.20-era APIs |
| `GAN/icgan/generate`, `GAN/icgan/guided` | Colab-notebook pastes: load SwAV/IC-GAN weights and loop over hardcoded datasets at import time |
| `style/omnimae` | loads `modelzoo/vitl_ssv2_ft.torch` at import time |
