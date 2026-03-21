# Maua Diffusion

Make visual art with diffusion models!

You can run these entry points either as modules (`python -m maua.diffusion.<module> ...`) or via the top-level CLI (`python -m maua diffusion <subcommand> ...`). The flags are the same; use `--help` on either form.

The `processors/` folder contains a number of wrappers around state-of-the-art diffusion models like Stable Diffusion, OpenAI's Guided Diffusion, the secondary model method a.k.a. Disco Diffusion, and more!

## Image Synthesis

`image.py` allows you to generate images using any of the implemented diffusion models.
It also contains functionality for progressively increasing the size of the image as well as scaling to arbitrary sizes through automatic tiling.
See `python -m maua.diffusion.image --help` or `python -m maua diffusion image --help` for information on all of the parameters.

Generating a 1024px image with GLID3-XL.

```bash
python -m maua diffusion image \
    --init path/to/an/interesting/starting/image.jpg \
    --text "A beautiful, well-thought-out prompt with extra vitamins" \
    --diffusion glid3xl \
    --cfg-scale 10 \
    --sizes 256,256 1024,1024 \
    --skips 0.5 0.8 \
    --stitch \
    --tile-size 256
```

## Video Stylization

`video.py` supports styling videos, keeping coherence over time by using optical flow (similar to [Disco Diffusion Warp by @sxela](https://github.com/Sxela/DiscoDiffusion-Warp)).
This script also supports all of the diffusion models.
See `python -m maua.diffusion.video --help` or `python -m maua diffusion video --help` for information on all of the parameters.

Stylizing a video with a text and image using Disco-like secondary model diffusion.

```bash
python -m maua diffusion video \
    --init path/to/a/video/with/cool/movement.mp4 \
    --text "An epic prompt made by a zombie unicorn" \
    --style path/to/an/image/with/a/cool/style.png \
    --diffusion guided \
    --guidance-speed fast \
    --clip-scale 2500 \
    --style-scale 750 \
    --color-match-scale 1500 \
    --blend 20 \
    --consistency-trust 0.5 \
    --size 512,512 \
    --first-skip 0.2 \
    --skip 0.7
```

Stylizing a video with Stable Diffusion.

```bash
python -m maua diffusion video \
    --init path/to/a/video/with/cool/movement.mp4 \
    --text "Some poem written by me" \
    --diffusion stable \
    --blend 50 \
    --consistency-trust 0.125 \
    --size 512,512 \
    --first-skip 0.2 \
    --skip 0.8 \
    --flow-exaggeration 5 \
    --sharpness 2 \
    --hist-persist
```

More advanced tutorials will be added soon.

## Fine-tuning Stable Diffusion

Requires 24 GB of VRAM at the moment :/

Currently only supports fine-tuning on a directory of images.
This has the effect of making all generated images tend towards the style of the supplied images.

The checkpoints that are saved by this script are compatible with the above scripts.

```bash
python -m maua diffusion finetune-stable \
    --datadir /path/to/images/folder/ \
    --resume modelzoo/stable-diffusion-v1-4.ckpt \
    --logdir modelzoo/ \
    --gpus 0,
```

## Hugging Face temporal ControlNet video (`temporalvideo_hf`)

Stylizes a video using diffusers ControlNet + optical flow warping. See `python -m maua diffusion video-hf --help`.

```bash
python -m maua diffusion video-hf -i input.mp4 -p "your prompt"
```

## KLMC2 animation (`klmc2_animation`)

Second-order Langevin MCMC animation from a text prompt (k-diffusion). See `python -m maua diffusion klmc2 --help`.

```bash
python -m maua diffusion klmc2 "a prompt" --n 120 --fps 20
```
