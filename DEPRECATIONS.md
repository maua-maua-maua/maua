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
| `diffusion/interpolate.py` `slerp()` | written for 3-D `(batch, P, N)` inputs; `permute(2,0,1)` crashed on the 2-D `(P, N)` the interpolate path passes | rewrote to broadcast `t` over `(P, N)` → `(P, k, N)` |
| `diffusion/processors/latent.py` | `q_sample` gathered `t-1 = -1` (CUDA device-side assert) when starting from no added noise | `t.clamp(1, original_num_steps)` |
| `diffusion/video.py` `WriteThread` | `run()` was an infinite `while True` with no exit; `finalize()`'s `join()` blocked forever | poison-pill: `finalize()` enqueues `(None, None)`, `run()` breaks on it |
| `diffusion/video.py` `FramesOnDisk.insert` | incremented `length` on every call, so overwrites (explicit `idx`) inflated `len()` past the on-disk frame count | `length = max(length, idx + 1)` |
| `diffusion/video.py` loop-fade | `loop_fade` is empty when `wrap_around=0`, but the loop runs one turbo-step past `N` and indexed it (`IndexError`) | guard the fade branch with `wrap_around > 0` |
| `diffusion/interp_loop.py` | script-style, unfinished experiment | wrapped in `main()` + guard, documented |
| `audiovisual/audioreactive/{audio,mir}.py`, `.../selfsupervised/sample.py` | `torchaudio.load` routes through torchcodec, whose native FFmpeg shim won't load on this stack; librosa 0.11 keyword renames (`rms(y=)`, `get_duration(path=)`) | decode via `librosa.load(sr=None, mono=True)`; added `separate_sources` wrapper over `unmix` |
| `audiovisual/generate.py` | `@torch.inference_mode()` around the lazy StyleGAN/NV-op import made `bias_act._null_tensor` an inference tensor, poisoning any later autograd graph (e.g. the GAN projector) | `@torch.no_grad()` |
| `audiovisual/patches/base/stylegan{2,3}.py`, `generate.py` | `resize_strategy` default `"pad-zero"` failed the `pad-{how}-{where}` parser | default `"stretch"` |
| `flow/consistency.py`, `flow/lib.py`, `style/video.py` | flow backends return a 3-D `(H,W,2)` map / tensor where numpy 4-D was assumed; `reliable` mask stored with an extra dim | coerce to batched tensors, `flow_warp_map(size=...)` resize, normalize `reliable` to `(1,H,W)` per frame |
| `flow/lib.py` `flow_warp_map` | module-level `NEUTRAL` grid cache was device-sticky (`.to(flow)` only on first build); a grid cached on cpu (under a leaked default-device mode) then added to a cuda flow put the sampling grid on the wrong device | rebuild `NEUTRAL` when `flow.device` changes |
| `ops/video.py` `WriteWorker` | writer thread self-terminated after 30 s of an empty queue — a slow first frame (compiling the StyleGAN CUDA plugins on first use) killed it before any frame was written, so ffmpeg finalized a video-stream-less file (`audio2video` `StopIteration` in `ffmpeg.probe`) | drop the 30-poll auto-exit; stop only on the explicit `stopping` flag, and `join()` the writer before closing ffmpeg's stdin so the last frame isn't truncated |
| `tests/conftest.py` (isolation) | `nca/{train,generate}.py`'s `set_default_device('cuda')` "restores" to the read-back value, which on a fresh process installs a cpu default-device *mode* that never existed, leaking into later tests; `rife.py` similarly leaks `set_default_tensor_type` | autouse fixture snapshots dtype and hard-resets `set_default_device(None)` + dtype after every test |
| `ops/io.py` | numpy 2.x raises `OverflowError` on `uint8` scalar overflow in the image hash | `int(ch) * 997` |
| `super/image/models/latent_diffusion.py` | tiling guard checked *image* size (≥128) but the fold runs on the *latent* (image/vqf) with a 128 kernel → 0 patches at 256px; torch 2.6+ `weights_only` broke the fully-pickled ckpt | threshold `128*vqf` (512px); `weights_only=False` |
| `GAN/projector.py` | hardcoded `/home/hans` paths, script-style; hardcoded `num_ws=18` and `size=1024` mismatched smaller generators | `project(model_file, file, out_dir, ..., steps)`; `num_ws`/`size` derived from the generator |
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

## needs-dep (continued) / import-wiring

| Module | Blocker |
|---|---|
| `GAN/ZSSGAN`, `GAN/nada` | The `maua.GAN.pix2pix` submodule is now present, but ZSSGAN's vendored code uses bare `from models import ...` / `from util import ...` that assume the pix2pix directory is on `sys.path`. Needs an import shim (add pix2pix to `sys.path`, or rewrite the imports to `maua.GAN.pix2pix.*`) before it will load. |

## Fragile submodule working-tree patches

Several vendored submodules only work with local working-tree edits that are **not**
committed to this repo (the parent repo only tracks the submodule commit pointer).
A `git submodule update` resets them and reintroduces the breakage. Known patches:

| Submodule | Patch |
|---|---|
| `submodules/VQGAN` | `taming/models/*`, LPIPS/vqperceptual tweaks (pre-existing) |
| `submodules/latent_diffusion` | `ldm/util.py`: a botched `print(...)` removal left a dangling f-string → `IndentationError`; repaired to a comment. Reintroduced when the submodule was reset during the pix2pix add. |
| `submodules/{BSRGAN,liteflownet,pwc,spynet,unflow}` | pre-existing local edits |
| `GAN/nv/torch_utils/custom_ops.py` | NVIDIA's `get_plugin` assumed `cpp_extension.load()` puts the build dir on `sys.path` and then `import_module(name)`; modern torch returns the compiled module directly, so the import failed (`bias_act_plugin` etc. `ModuleNotFoundError`). Patched to use the `load()` return value, falling back to `import_module` only if it's `None`. |

The `torch._six` shim in `maua/ops/compat.py` covers VQGAN/taming's
`from torch._six import string_classes` so that particular breakage no longer depends
on a working-tree patch.

## Audio-reactive example patches

The `audioreactive` library was refactored (features no longer take `n_frames`/`margin`/
`clip`/`smooth` — you `resample()` separately; `rms`→`volume`; `chroma_weight_latents`
removed). The old `patches/examples/{stylegan2,stylegan3}.py` still use the pre-refactor
API and need porting (quarantine candidates for Phase C). A new minimal
`patches/examples/simple.py` was written against the current API and is what the
`audio2video` smoke test loads.

## quarantine (Phase C → `maua/legacy/`)

| Module | Why |
|---|---|
| `autoregressive/min_dalle/generate` | minDALL-E submodule uses mutable dataclass defaults, rejected by Python ≥3.12 (would need forking upstream) |
| `autoregressive/rq_dalle` | rqvae submodule, same mutable-dataclass-default problem |
| `autoregressive/ru_dalle` (`api`, `generate`, `finetune`) | `rudalle` needs `huggingface_hub.cached_download`, removed from modern `huggingface_hub` |
| `autoregressive/cog/video/{generate,infinite}` | CogVideo / `icetk` need protobuf<3.20-era APIs |
| `GAN/icgan/generate`, `GAN/icgan/guided` | Colab-notebook pastes: load SwAV/IC-GAN weights and loop over hardcoded datasets at import time |
| `style/omnimae` | loads `modelzoo/vitl_ssv2_ft.torch` at import time |
