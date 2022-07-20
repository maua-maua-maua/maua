# fmt:off
import sys

import decord
import numpy as np
import torch
from tqdm import tqdm

from ..ops.video import write_video
from .conditioning import (CLIPGrads, ColorMatchGrads, ContentPrompt,
                           LPIPSGrads, StylePrompt, TextPrompt, VGGGrads)
from .loop import (VideoFrames, initialize_cache_files,
                   initialize_optical_flow, warp)
from .sample import build_output_name, round64
from .wrappers.glide import GLIDE
from .wrappers.guided import GuidedDiffusion
from .wrappers.latent import LatentDiffusion

# fmt:on

decord.bridge.set_bridge("torch")


if __name__ == "__main__":
    W, H = 256, 256
    timesteps = 100
    skip = 0.4
    blend_every = None
    blend = 2
    consistency_trust = 0.75
    text = sys.argv[2]
    init = sys.argv[1]
    style_img = None
    fps = 12
    cfg_scale = 3
    clip_scale = 2500
    lpips_scale = 2000
    style_scale = 0
    color_match_scale = 0
    diffusion_speed = "fast"
    diffusion_sampler = "ddim"
    turbo = 1
    diffusion_model = "large"
    device = "cuda"

    # process user inputs
    W, H = round64(W), round64(H)
    n_steps = round((1 - skip) * timesteps)
    blend_every = (
        round((1 - skip) * timesteps)
        if blend_every is None
        else (round(blend_every * timesteps) if blend_every < 1 else blend_every)
    )

    # build output name based on inputs
    out_name = build_output_name(init, style_img, text)

    # initialize cache files
    cache = initialize_cache_files(out_name, H, W, device)

    # load init video
    content = VideoFrames(init, H, W, device)

    # initialize_optical_flow(cache, content)
    # neutral = torch.stack(torch.meshgrid(torch.linspace(-1, 1, W), torch.linspace(-1, 1, H), indexing="xy"), axis=2)
    # print("                  ", "min     ", "mean     ", "max     ", "shape")
    # forward = cache.forward[list(range(len(cache.forward)))] - neutral[None].to(device)
    # print("forward flow (px):", forward.min().item(), forward.mean().item(), forward.max().item(), forward.shape)
    # write_video(
    #     torch.stack([torch.from_numpy(flow_to_image(f.cpu().numpy())) for f in forward]).permute(0, 3, 1, 2).div(255),
    #     f"output/{Path(out_name).stem}_forward_flow.mp4",
    # )
    # backward = cache.backward[list(range(len(cache.backward)))] - neutral[None].to(device)
    # print("backward flow (px):", backward.min().item(), backward.mean().item(), backward.max().item(), backward.shape)
    # write_video(
    #     torch.stack([torch.from_numpy(flow_to_image(f.cpu().numpy())) for f in backward]).permute(0, 3, 1, 2).div(255),
    #     f"output/{Path(out_name).stem}_backward_flow.mp4",
    # )
    # reliable = cache.reliable[list(range(len(cache.reliable)))]
    # print("reliable flow (0,1):", reliable.min().item(), reliable.mean().item(), reliable.max().item(), reliable.shape)
    # write_video(reliable.tile(1, 3, 1, 1), f"output/{Path(out_name).stem}_reliable_flow.mp4")
    # exit()

    # initialize diffuser
    # diffusion = GuidedDiffusion(
    #     [
    #         CLIPGrads(scale=clip_scale),
    #         LPIPSGrads(scale=lpips_scale),
    #         VGGGrads(scale=style_scale),
    #         ColorMatchGrads(scale=color_match_scale),
    #     ],
    #     sampler=diffusion_sampler,
    #     timesteps=timesteps,
    #     model_checkpoint=diffusion_model,
    #     speed=diffusion_speed,
    # ).to(device)
    # diffusion = LatentDiffusion(
    #     sampler=diffusion_sampler,
    #     timesteps=timesteps,
    #     model_checkpoint=diffusion_model,
    # ).to(device)
    diffusion = GLIDE(sampler=diffusion_sampler, timesteps=timesteps).to(device)

    start_idx, direction = 0, 1
    total_steps = len(content) * n_steps
    with tqdm(total=total_steps) as progress, torch.no_grad():
        for step in range(0, n_steps, blend_every):
            progress.set_description(f"Step {step + 1} - {step + blend_every} of {n_steps}")

            if step == 0:
                # in the first step we initialize with the content video
                frames = content

                # we also need to calculate optical flows
                initialize_optical_flow(cache, frames)
            else:
                # load init images for this pass
                frames = cache.old

            N = len(frames)

            with cache.new:
                frame_range = np.arange(N) if direction > 0 else np.flip(np.arange(N))
                roll = np.random.randint(1, N)
                frame_range = np.roll(frame_range, roll)
                start_idx = frame_range[0]
                for f_i, f_n in enumerate(frame_range):
                    progress.update(blend_every)

                    if f_i % turbo != 0:
                        out_img = warp(out_img, (cache.forward if direction == 1 else cache.backward)[f_n])
                        cache.new.append(out_img)
                        continue

                    init_img = frames[f_n]

                    if blend > 0:
                        flow_mask = cache.reliable[f_n]
                        flow_mask *= consistency_trust
                        flow_mask += 1 - consistency_trust
                        flow_mask *= blend

                        flow = (cache.forward if direction == 1 else cache.backward)[f_n]

                        prev_img = frames[(f_n - direction) % N] if f_i == 0 else out_img

                        init_img += flow_mask * warp(prev_img, flow)
                        init_img /= 1 + flow_mask

                    prompts = [ContentPrompt(content[(start_idx + f_n * direction) % len(content)])]
                    if text is not None:
                        prompts.append(TextPrompt(text, weight=cfg_scale))
                    if style_img is not None:
                        prompts.append(StylePrompt(path=style_img, size=(H, W)))

                    out_img = diffusion.sample(
                        init_img, prompts=prompts, start_step=n_steps - step, n_steps=blend_every, verbose=False
                    )
                    cache.new.append(out_img)

            write_video(
                np.load(cache.new.file, mmap_mode="r") * 0.5 + 0.5, f"output/{out_name}_{step + 1}.mp4", fps=fps
            )
            cache.old.update(cache.new)
            direction = -direction  # reverse direction of flow weighting

    write_video(np.load(cache.old.file, mmap_mode="r") * 0.5 + 0.5, f"output/{out_name}.mp4", fps=fps)
