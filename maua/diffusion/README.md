# Maua Diffusion

Make visual art with diffusion models!

The `processors/` folder contains a number of wrappers around state-of-the-art diffusion models like Stable Diffusion, OpenAI's Guided Diffusion, the secondary model method a.k.a. Disco Diffusion, and more!

## Image Synthesis

`image.py` allows you to generate images using any of the implemented diffusion models.
It also contains functionality for progressively increasing the size of the image as well as scaling to arbitrary sizes through automatic tiling.
See `python -m maua.diffusion.image --help` for information on all of the parameters.

```
  --init INIT           How to initialize the image "random", "perlin", or a path to an
                        image file. (default: random)
  --text TEXT           A text prompt to visualize. (default: None)
  --content CONTENT     A content image whose structure to adapt in the output image
                        (only works with "guided" diffusion at the moment, see --lpips-
                        scale). (default: None)
  --style STYLE         An image whose style should be optimized for in the output
                        image (only works with "guided" diffusion at the moment, see
                        --style-scale). (default: None)
  --sizes SIZES [SIZES ...]
                        Sequence of sizes to synthesize the image at. (default: (512,
                        512))
  --skips SKIPS [SKIPS ...]
                        Sequence of skip fractions for each size. Lower fractions will
                        stray further from the original image, while higher fractions
                        will hallucinate less detail. (default: 0)
  --timesteps TIMESTEPS
                        Number of timesteps to sample the diffusion process at. Higher
                        values will take longer but are generally of higher quality.
                        (default: 50)
  --super-res SUPER_RES
                        Super resolution model to upscale intermediate results with
                        before applying next diffusion resolution (see maua.super.image
                        --model-help for full list of possibilities, None to perform
                        simple resizing). (default: SwinIR-M-DFO-GAN)
  --stitch              Enable tiled synthesis of images which are larger than the
                        specified --tile-size. (default: False)
  --tile-size TILE_SIZE
                        The maximum size of tiles the image is cut into. (default:
                        None)
  --max-batch MAX_BATCH
                        Maximum batch of tiles to synthesize at one time (lower values
                        use less memory, but will be slower). (default: 4)
  --diffusion {guided,latent,glide,glid3xl,stable}
                        Which diffusion model to use. (default: stable)
  --sampler {p,ddim,plms,euler,euler_ancestral,heun,dpm_2,dpm_2_ancestral,lms}
                        Which sampling method to use. "p", "ddim", and "plms" work for
                        all diffusion models, the rest are currently only supported
                        with "stable" diffusion. (default: dpm_2)
  --guidance-speed {regular,fast}
                        How to perform "guided" diffusion. "regular" is slower but can
                        be higher quality, "fast" corresponds to the secondary model
                        method (a.k.a. Disco Diffusion). (default: fast)
  --clip-scale CLIP_SCALE
                        Controls strength of CLIP guidance when using "guided"
                        diffusion. (default: 2500.0)
  --lpips-scale LPIPS_SCALE
                        Controls the apparent influence of the content image when using
                        "guided" diffusion and a --content image. (default: 0.0)
  --style-scale STYLE_SCALE
                        When using "guided" diffusion and a --style image, a higher
                        --style-scale enforces textural similarity to the style, while
                        a lower value will be conceptually similar to the style.
                        (default: 0.0)
  --color-match-scale COLOR_MATCH_SCALE
                        When using "guided" diffusion, the --color-match-scale guides
                        the output's colors to match the --style image. (default: 0.0)
  --cfg-scale CFG_SCALE
                        Classifier-free guidance strength. Higher values will match the
                        text prompt more closely at the cost of output variability.
                        (default: 7.5)
  --match-hist          Match the histogram of the initialization image to the --style
                        image before starting diffusion. (default: False)
  --sharpness SHARPNESS
                        Sharpen the image by this amount after each diffusion scale (a
                        value of 1.0 will leave the image unchanged, higher values will
                        be sharper). (default: 0.0)
  --device DEVICE       Which device to use (e.g. "cpu" or "cuda:1") (default: cuda)
  --out-dir OUT_DIR     Directory to save output images to. (default: output/)
```

Generating a 1024px image with GLID3-XL.

```bash
python -m maua.diffusion.image \
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

```
  --init INIT           How to initialize the image "random", "perlin", or a path to an
                        image file. (default: random)
  --text TEXT           A text prompt to visualize. (default: None)
  --style STYLE         An image whose style should be optimized for in the output
                        image (only works with "guided" diffusion at the moment, see
                        --style-scale). (default: None)
  --size SIZE           Size to synthesize the video at. (default: (512, 512))
  --skip SKIP           Lower fractions will stray further from the original image,
                        while higher fractions will hallucinate less detail. (default:
                        0.7)
  --first-skip FIRST_SKIP
                        Separate skip fraction for the first frame. (default: 0.4)
  --timesteps TIMESTEPS
                        Number of timesteps to sample the diffusion process at. Higher
                        values will take longer but are generally of higher quality.
                        (default: 50)
  --blend BLEND         Factor with which to blend previous frames into the next frame.
                        Higher values will stay more consistent over time (e.g. --blend
                        20 means 20:1 ratio of warped previous frame to new input
                        frame). (default: 2)
  --consistency-trust CONSISTENCY_TRUST
                        How strongly to trust flow consistency mask. Lower values will
                        lead to more consistency over time. Higher values will respect
                        occlusions of the background more. (default: 0.75)
  --wrap-around WRAP_AROUND
                        Number of extra frames to continue for, looping back to start.
                        This allows for seamless transitions back to the start of the
                        video. (default: 0)
  --turbo TURBO         Only apply diffusion every --turbo'th frame, otherwise just
                        warp the previous frame with optical flow. Can be much faster
                        for high factors at the cost of some visual detail. (default:
                        1)
  --flow-exaggeration FLOW_EXAGGERATION
                        Factor to multiply optical flow with. Higher values lead to
                        more extreme movements in the final video. (default: 1)
  --diffusion {guided,latent,glide,glid3xl,stable}
                        Which diffusion model to use. (default: stable)
  --sampler {p,ddim,plms,euler,euler_ancestral,heun,dpm_2,dpm_2_ancestral,lms}
                        Which sampling method to use. "p", "ddim", and "plms" work for
                        all diffusion models, the rest are currently only supported
                        with "stable" diffusion. (default: dpm_2)
  --guidance-speed {regular,fast}
                        How to perform "guided" diffusion. "regular" is slower but can
                        be higher quality, "fast" corresponds to the secondary model
                        method (a.k.a. Disco Diffusion). (default: fast)
  --clip-scale CLIP_SCALE
                        Controls strength of CLIP guidance when using "guided"
                        diffusion. (default: 2500.0)
  --lpips-scale LPIPS_SCALE
                        Controls the apparent influence of the content image when using
                        "guided" diffusion and a --content image. (default: 0.0)
  --style-scale STYLE_SCALE
                        When using "guided" diffusion and a --style image, a higher
                        --style-scale enforces textural similarity to the style, while
                        a lower value will be conceptually similar to the style.
                        (default: 0.0)
  --color-match-scale COLOR_MATCH_SCALE
                        When using "guided" diffusion, the --color-match-scale guides
                        the output's colors to match the --style image. (default: 0.0)
  --cfg-scale CFG_SCALE
                        Classifier-free guidance strength. Higher values will match the
                        text prompt more closely at the cost of output variability.
                        (default: 7.5)
  --match-hist          Match the color histogram of the initialization image to the
                        --style image before starting diffusion. (default: False)
  --hist-persist        Match the color histogram of subsequent frames to the first
                        diffused frame (helps alleviate oversaturation). (default:
                        False)
  --sharpness SHARPNESS
                        Sharpen the image by this amount after each diffusion scale (a
                        value of 1.0 will leave the image unchanged, higher values will
                        be sharper). (default: 0.0)
  --constant-seed CONSTANT_SEED
                        Use a fixed noise seed for all frames (None to disable).
                        (default: None)
  --device DEVICE       Which device to use (e.g. "cpu" or "cuda:1") (default: cuda)
  --preview             Show frames as they're rendered (moderately slower). (default:
                        False)
  --fps FPS             Framerate of output video. (default: 12)
  --out-dir OUT_DIR     Directory to save output images to. (default: output/)
```

Stylizing a video with a text and image using Disco-like secondary model diffusion.

```bash
python -m maua.diffusion.video \
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
python -m maua.diffusion.video \
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
    --hist-presist
```

More advanced tutorials will be added soon.
