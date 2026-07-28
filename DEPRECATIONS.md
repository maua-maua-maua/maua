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

**Env footgun (ninja):** `~/.local/bin/ninja` is a broken stub (its `#!/usr/bin/python3`
shebang points at a system python without the `ninja` module). If the conda env's `bin`
is not ahead of `~/.local/bin` on PATH, torch `cpp_extension.load()` finds the broken
stub, `verify_ninja_availability()` fails, and every StyleGAN-CUDA-op import
(ZSSGAN/nada, GAN/nv plugins) breaks. Run tests with
`PATH=/home/jcbgb/anaconda3/envs/maua/bin:$PATH` (ninja is pip-installed in the env and
pinned in setup.py).

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

The rescued Colab scripts also route their weights through `maua/ops/download.py`:
IC-GAN's SwAV extractor (`swav_pretrained.pth.tar`) and OmniMAE's
`vitl_ssv2_ft.torch` are fetched via `fetch_model(..., url=<fbaipublicfiles>)` — still
live upstream, so they use the URL fallback rather than the HF mirror for now.

Two more dead hosts surfaced while broadening the style-transfer smoke tests, both now
resolved: `mirror.io.community` VQGAN weights (imagenet_1024/16384 repointed to the
original CompVis heibox share, verified live; wikiart_1024/16384 mirrors are still all
dead — mirror.io.community and eaidata.bmk.sh both gone — provide those in `modelzoo/`
manually) and `web.eecs.umich.edu` (ProGamerGov `vgg16/vgg19` for the `pgg-*`
perceptors, repointed to the HF mirror `AfrodreamsAI/afrodreams`; the vgg19 path was
proven end-to-end from an empty modelzoo).

---

## Audio tree rescue (post-Phase-B review)

`maua/audio/**` (99 modules: jukebox, RAVE, neural waveshaping, granular, selfsupervised
sample tooling) was excluded from the fast import test wholesale during Phase B. A
follow-up review rescued the entire tree — all 99 modules now import cleanly and are
part of the fast import surface (the `audio` EXCLUDE entry was removed):

| Module | What broke | Fix |
|---|---|---|
| `audio/{jukebox,waveshaping}/**`, `audio/rave/**` | vendored from repos where `jukebox`/`waveshaping`/`rave`/`prior` were top-level packages; bare intra-package imports failed | scoped `sys.path` shims in `maua/audio/__init__.py` and `maua/audio/rave/__init__.py` (same pattern as pix2pix/ZSSGAN) |
| rave/waveshaping/jukebox deps | `cached_conv`, `udls`, `gin-config`, `fire`, `mpi4py` not installed | added to `setup.py` install_requires (all resolve on the modern stack) |
| `audio/rave/rave/pqmf.py` | scipy ≥1.15 removed `firwin(nyq=)` and the `scipy.signal.kaiser` re-export; `fmin` explores out-of-range cutoffs that new firwin hard-errors on | `fs=2*np.pi` equivalent, dropped unused imports, penalty-guard in `loss_wc`; PQMF forward/inverse roundtrip numerically verified (err ~2.5e-3) |
| `audio/rave/{combine_models,export_rave,export_prior,train_prior}.py` | script-style: module-level `args.parse_args()` + model loads at import | wrapped in `main()` + `__main__` guards; rave/prior imports deferred where cached_conv buffer mode must be set first |
| `audio/jukebox/{Interacting_with_Jukebox,tests/test_sample}.py` | Colab paste executing at import; `check_sample()` ran at import (needs distributed init) | wrapped behind `main()` / `__main__` guards |
| `audio/waveshaping/scripts/resynthesise_dataset.py` | imported `URMPDataset`, which the vendored copy never included | aliased to `GeneralDataset` (same on-disk format) |
| `audio/waveshaping/scripts/NEWT_Timbre_Transfer.py` | `google.colab` import; module-level notebook cells | try/except colab import; cells wrapped in `main()` with colab-only cells guarded |

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
| `flow/sniklaus.py` neural backends (pwc/liteflownet/unflow) | need a runtime-compiled CUDA correlation kernel via cupy, which was uninstalled; the vendored kernels also used `cupy.cuda.compile_with_cache`, removed in cupy 13 | install `cupy-cuda12x<14` (numpy-1.26-compatible); `sniklaus.py` idempotently rewrites the kernels to `cupy.RawModule` at import. spynet is pure-torch (no cupy). All four + farneback covered by `tests/smoke/test_flow_models.py` |
| `diffusion/{image,video,klmc2_animation}.py` `main()` | splatted argparse's `set_defaults(func=...)` dispatch handle into the sampler → `unexpected keyword argument 'func'`; image `main` also never created `--out-dir` | drop `func` before splatting; `mkdir(parents=True)` the out-dir. Covered by `tests/smoke/test_cli_e2e.py` |
| `GAN/blending.py` `blend_checkpoints` | called `get_state_dict_key_levels` unconditionally; its module-name parser IndexErrors on the current wrapper naming, but the levels are only used by the "crossover" strategy | compute levels lazily (crossover only), so the "random" model-soup path works |
| `style/image_multires.py` `transfer_multires` | wrote intermediates to a `sys.argv`-derived path that crashes when called off the CLI | added a `save_intermediate` arg (None disables saving) |
| `GAN/ZSSGAN` (StyleGAN-NADA) | five modern-stack breakages surfaced by end-to-end `test_nada`: (1) torch 2.6+ `weights_only=True` default rejected the pickled NVIDIA ckpt; (2) `torch.load` can't read an NVIDIA distribution `.pkl` (raw pickle, not a torch archive); (3) `SG2Generator` hardcoded `channel_multiplier=2`, mismatching the half-width FFHQ-256 research pkl; (4) this pix2pix vintage's `define_G` dropped the trailing `gpu_ids` arg; (5) torchvision `save_image` renamed `range=`→`value_range=` | (1) `weights_only=False`; (2) new `load.py:load_nvidia_state_dict` routes `.pkl` through NVIDIA's `legacy.load_network_pkl(...).state_dict()`; (3) new `load.py:detect_channel_multiplier` reads the multiplier back from the res-64 block width; (4) dropped the `[0]` arg; (5) `value_range=` |
| `audiovisual/patches/**`, `audiovisual/patches/primitives/latents.py` | stale relative-ish imports after renames | absolute imports |
| torchvision compat | `torchvision.models.utils` & `transforms.functional_tensor` removed (needed by basicsr/ic_gan/submodules) | sys.modules shims in `maua/ops/compat.py`, installed from `maua/__init__.py` |
| `transformers.top_k_top_p_filtering` | removed from `transformers` (~4.39); the ru_dalle submodule's sampler still calls it | reinstated the canonical implementation as an attribute on `transformers` in `maua/ops/compat.py` |
| dep pins | `pkg_resources` & PL 1.x APIs | `setuptools<81`, `pytorch-lightning<2` in `setup.py` |
| `workspace/examples/audio/*.mp3` | committed files were truncated/corrupt | replaced with clean full-length stream copies |

## Absorbed research dependencies (vendored into the tree)

Several modules depended on research code that isn't installable on a modern stack (no
PyPI wheel, or unsatisfiable pins). Rather than drop them, the handful of functions
actually used were vendored into local `_name.py` modules with provenance headers and
the imports repointed at the vendored copy. All import cleanly and are covered by the
fast import test.

| Module | Was | Now |
|---|---|---|
| `.../selfsupervised/features/correlation` | `from anatome.distance import ...`; `from torchsort import soft_rank` | vendored `_anatome_distance.py` (moskomule/anatome, Apache-2.0) + `_soft_rank.py` (google-research fast-soft-sort, pure torch/numpy; correctness-checked) |
| `.../selfsupervised/features/efficient_quantile` | C++ extension built in place | reimplemented pure-torch (`torch.quantile`, `kthvalue` for large tensors) |
| `GAN/training/models/experimental/deepinvolutional` | `from involution import Involution2d` | vendored `_involution.py` (ChristophReich1996/Involution) |
| `GAN/training/models/experimental/stylehypermixerfly` | `from torch_butterfly import Butterfly, ...` | vendored `_torch_butterfly.py` (HazyResearch/fly, Apache-2.0; real-only subset, forward/backward-checked) |

## Now pip-installable (dep resolved on the modern stack)

| Module | Dep |
|---|---|
| `GAN/training/models/experimental/equivariant` | `escnn` (installs; pulls numpy 1.26 — benign tension with opencv's numpy≥2 want) |
| `.../selfsupervised/features/correlation` | `torchmetrics` (for `matthews_corrcoef`) |
| `autoregressive/cog/video/{generate,infinite}` | `SwissArmyTransformer` (installs as `sat` on modern PyPI; the old top-level name is aliased in `compat.py`) |

## Colab-script rescues (module-level execution wrapped behind `main()`)

These were Colab-notebook pastes that loaded weights or read hardcoded `/home/hans`
paths at import time. Each was wrapped in a callable + `argument_parser()` + `main(args)`
+ `__main__` guard (the `main(args)` convention the CLI's lazy dispatch expects), with
weights routed through `maua/ops/download.py`. All now import cleanly.

| Module | Blocked import on | Now |
|---|---|---|
| `GAN/nada` | `Image.open("/home/hans/...")` + hardcoded paths | `train(...)` + argparse; ZSSGAN import wired (see below) |
| `GAN/icgan/generate` | loaded SwAV/IC-GAN weights + looped a hardcoded dataset glob | `main(args)`; SwAV via `fetch_model`; `--input-dir`/`--output-dir` |
| `GAN/icgan/guided` | star-imports `generate` | safe now that `generate` is import-clean |
| `style/omnimae` | loaded `vitl_ssv2_ft.torch` | `style_transfer(...)` + argparse; ckpt via `fetch_model` |
| `autoregressive/cog/video/generate` | icetk/protobuf + SwissArmyTransformer (see compat.py) | argparse `main`; import-clean |

## Import-wiring fixes

| Module | Fix |
|---|---|
| `GAN/ZSSGAN`, `GAN/nada` | ZSSGAN's vendored code does bare `from models import ...` expecting the pix2pix dir on `sys.path`. `ZSSGAN.py` now prepends `maua/GAN/pix2pix` to `sys.path` before importing, and imports the vendored CycleGAN networks as `maua.GAN.pix2pix.models.networks`. |
| `autoregressive/{min_dalle,rq_dalle,ru_dalle}` | minDALL-E/rqvae mutable-dataclass-default crash patched in the submodules (see fragile patches); `ru_dalle` uses the `huggingface_hub.cached_download` shim in `compat.py`. |
| `autoregressive/cog/video/{generate,infinite}` | CogVideo does bare `from models import ...`, but its `models` is a *namespace* package that loses the name to pix2pix's *regular* `models` package (`models/__init__.py`) whenever pix2pix is imported anywhere in the same process (e.g. the fast import walk), regardless of `sys.path` order. `_import_cogvideo_globals()` resolves the three bare imports with pix2pix's root temporarily removed from `sys.path` and the colliding cached `models`/`coglm_strategy`/`sr_pipeline` dropped, then restores both. Also: both modules add the same special tokens to the shared `icetk.icetk` singleton, so the `add_special_tokens` call is guarded to no-op on the "already defined" re-add. |

## compat.py shims added for CogVideo

`maua/ops/compat.py:install_shims()` (run from `maua/__init__.py`) gained two entries so
CogVideo imports without an environment-variable dance or a global pure-Python protobuf
penalty:
- **icetk stale protobuf**: icetk ships a `sentencepiece_model_pb2` generated against an
  ancient protobuf the modern runtime rejects. The `sentencepiece` package's up-to-date,
  API-compatible equivalent is pre-injected as `sys.modules["icetk.sentencepiece_model_pb2"]`
  so icetk never loads its stale file (works with either protobuf backend).
- **SwissArmyTransformer→sat**: aliased in `sys.modules` so the old top-level import name
  resolves to the renamed `sat` package.

## Intentionally dropped

Per the repo owner, these optional deps are not worth vendoring; the modules that need
them are the **only remaining `maua/legacy/` candidates** for Phase C.

| Dep | Modules | Why dropped |
|---|---|---|
| `padl` | `GAN/training/train_v0` | abandoned upstream |
| `ffcv` | `GAN/training/trainer`, `GAN/training/dataset/image` | unmaintained, compile-heavy |

## Fragile submodule working-tree patches

Several vendored submodules only work with local working-tree edits that are **not**
committed to this repo (the parent repo only tracks the submodule commit pointer).
A `git submodule update` resets them and reintroduces the breakage. Known patches:

| Submodule | Patch |
|---|---|
| `submodules/VQGAN` | `taming/models/*`, LPIPS/vqperceptual tweaks (pre-existing) |
| `submodules/latent_diffusion` | `ldm/util.py`: a botched `print(...)` removal left a dangling f-string → `IndentationError`; repaired to a comment. Reintroduced when the submodule was reset during the pix2pix add. |
| `submodules/{BSRGAN,liteflownet,pwc,spynet,unflow}` | pre-existing local edits. Additionally, `sniklaus.py` now rewrites the `{pwc,liteflownet,unflow}/correlation/correlation.py` kernels from the removed `cupy.cuda.compile_with_cache` API to `cupy.RawModule` at **import time** — this patch is idempotent and driven from committed parent-repo code, so unlike the rows above it self-heals after a submodule reset. |
| `submodules/minDALLE` | `dalle/utils/config.py`: mutable dataclass default rejected by Python ≥3.12; changed to `field(default_factory=...)`. |
| `submodules/rq_vae_transformer` | `rqvae/models/rqtransformer/configs.py`: same mutable-dataclass-default fix (`field(default_factory=...)`). |
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

Following the "keep everything I possibly can" rescue pass, the previous quarantine
candidates — the DALL-E family (`min_dalle`, `rq_dalle`, `ru_dalle`), CogVideo, IC-GAN,
and OmniMAE — were all repaired and import cleanly (see the rescue tables above). The
**only** remaining legacy candidates are the three training-stack modules that depend on
the intentionally-dropped `padl`/`ffcv` (listed under *Intentionally dropped*). They stay
xfailed in `tests/fast/test_imports.py` until the Phase C `git mv` to `maua/legacy/`.

`maua/diffusion/flux2hd.py` (untracked) still loads the full FLUX pipeline at import; it
is wrapped + moved to `maua/text2img/flux.py` during Phase C rather than quarantined.
