import os
import shutil
from pathlib import Path
from uuid import uuid4

import decord
import numpy as np
import torch
from maua.flow.lib import flow_warp_map, get_consistency_map
from npy_append_array import NpyAppendArray as NpyFile
from resize_right import resize
from torch.nn.functional import grid_sample
from tqdm import tqdm

from ..audiovisual.audioreactive.signal import gaussian_filter
from ..flow import get_flow_model
from ..ops.video import write_video
from .conditioning import CLIPGrads, ContentPrompt, LPIPSGrads, TextPrompt
from .sample import round64
from .wrappers.guided import GuidedDiffusion

decord.bridge.set_bridge("torch")


def warp(x, f):
    return grid_sample(x, f, padding_mode="reflection", align_corners=False)


def load_from_memmap(mmap, idx, h, w):
    if not isinstance(idx, (list, np.ndarray)):
        idx = [idx]
    tensor = torch.from_numpy(mmap[idx].copy()).float().cuda()
    if tensor.dim() < 4:
        tensor = tensor.unsqueeze(1)
    if tensor.shape[2] != h or tensor.shape[3] != w:
        tensor = resize(tensor, out_shape=(h, w))
    return tensor


def flow_weighted(img, prev, flow, flow_mask, blend=1, consistency_trust=0.75):
    flow_mask *= consistency_trust
    flow_mask += 1 - consistency_trust
    flow_mask *= blend
    img += flow_mask * warp(prev, flow)
    img /= 1 + flow_mask
    return img


if __name__ == "__main__":
    W, H = 512, 512
    timesteps = 25
    skip = 13 / 25
    text = "a beautiful detailed ink illustration of a futuristic city skyline covered in irridescent windows and crystal glass, neo-tokyo metropolis"
    init = "/home/hans/modelzoo/diffusionGAN/denoising/take0/meer netsj interps/_diffusionGAN_interpolation_denoising_epoch82_seed48865_5065ae.mp4"
    init = "/home/hans/datasets/video/pleated.mp4"
    blend = 0.5
    consistency_trust = 0.75
    blend_every = 2
    fps = 18
    clip_scale = 8000
    lpips_scale = 1000
    diffusion_speed = "fast"
    diffusion_sampler = "plms"
    turbo_start = 8

    # process user inputs
    W, H = round64(W), round64(H)
    n_steps = round((1 - skip) * timesteps)
    turbo_schedule = [turbo_start / 2**i for i in range(round(np.log2(turbo_start)) + 1)]
    turbo_schedule = np.repeat(turbo_schedule, n_steps / blend_every // len(turbo_schedule)).astype(int)

    # initialize cache files
    out_name = f"{text.replace(' ','_')}"
    if init is not None:
        out_name = f"{Path(init).stem}_{out_name}"
    prev_frame_file, next_frame_file = f"workspace/{out_name}_frames_new.npy", f"workspace/{out_name}_frames_old.npy"
    fwd_flow_file, bkwd_flow_file = f"workspace/{out_name}_forward_flow.npy", f"workspace/{out_name}_backward_flow.npy"
    consistency_file = f"workspace/{out_name}_flow_consistency.npy"
    for file in [prev_frame_file, next_frame_file, fwd_flow_file, bkwd_flow_file, consistency_file]:
        if os.path.exists(file):
            os.remove(file)
    out_name += f"_{str(uuid4())[:6]}"

    # load init and models
    content = decord.VideoReader(init, width=W, height=H)
    flow_model = get_flow_model()
    diffusion = GuidedDiffusion(
        [CLIPGrads(scale=clip_scale), LPIPSGrads(scale=lpips_scale)],
        sampler=diffusion_sampler,
        timesteps=timesteps,
        speed=diffusion_speed,
    )

    start_idx, direction = 0, 1
    total_steps = sum([round(blend_every * len(content) / turbo) for turbo in turbo_schedule])
    with tqdm(total=total_steps) as progress, torch.no_grad():
        for step, turbo in zip(range(0, n_steps, blend_every), turbo_schedule):
            progress.set_description(f"Step {step + 1} / {n_steps}, Turbo {turbo}...")
            t_d = turbo * direction

            # load init images for this pass
            if os.path.exists(prev_frame_file):
                frames = np.load(prev_frame_file, mmap_mode="r")
            else:
                idxs = np.arange(0, len(content), turbo)
                frames = torch.stack([content[i].permute(2, 0, 1).div(127.5).sub(1) for i in idxs]).numpy()

            # in the first step we need to calculate some initial optical flows
            if step == 0:
                with NpyFile(fwd_flow_file) as forward, NpyFile(bkwd_flow_file) as backward, NpyFile(
                    consistency_file
                ) as consistency:
                    for f_n in range(frames.shape[0]):
                        prev = content[start_idx + (f_n - 1) * t_d].div(127.5).sub(1)
                        curr = content[start_idx + f_n * t_d].div(127.5).sub(1)
                        ff = flow_model(prev, curr)
                        bf = flow_model(curr, prev)
                        forward.append(np.ascontiguousarray(ff[None]))
                        backward.append(np.ascontiguousarray(bf[None]))
                        consistency.append(np.ascontiguousarray(get_consistency_map(ff, bf)[None]))
                forward = np.load(fwd_flow_file, mmap_mode="r")
                backward = np.load(bkwd_flow_file, mmap_mode="r")
                consistency = np.load(consistency_file, mmap_mode="r")

            # when the turbo schedule changes we double the temporal resolution ==> recalculate optical flow
            if len(content) / turbo > frames.shape[0]:

                # remove old cache, initialize new ones
                os.remove(fwd_flow_file), os.remove(bkwd_flow_file), os.remove(consistency_file)
                with NpyFile(next_frame_file) as new_frames, NpyFile(fwd_flow_file) as forward, NpyFile(
                    bkwd_flow_file
                ) as backward, NpyFile(consistency_file) as consistency:

                    # load a new frame halfway between each pair of
                    for f_n in range(frames.shape[0]):
                        prev = content[(start_idx + (f_n - 1) * t_d) % len(content)].div(127.5).sub(1)
                        btwn = content[(start_idx + round((f_n - 0.5) * t_d)) % len(content)].div(127.5).sub(1)
                        curr = content[(start_idx + f_n * t_d) % len(content)].div(127.5).sub(1)

                        ff1, ff2 = flow_model(prev, btwn), flow_model(btwn, curr)
                        bf1, bf2 = flow_model(btwn, prev), flow_model(curr, btwn)
                        fc1, fc2 = get_consistency_map(ff1, bf1), get_consistency_map(ff2, bf2)

                        prev = load_from_memmap(frames, (f_n - direction) % N, H, W)
                        curr = load_from_memmap(frames, f_n, H, W)
                        new = (warp(prev, flow_warp_map(ff1, (H, W))) + warp(curr, flow_warp_map(bf1, (H, W)))) / 2

                        new_frames.append(torch.cat((new, curr)).cpu().contiguous().numpy())
                        forward.append(np.ascontiguousarray(np.stack((ff1, ff2))))
                        backward.append(np.ascontiguousarray(np.stack((bf1, bf2))))
                        consistency.append(np.ascontiguousarray(np.stack((fc1, fc2))))

                shutil.move(next_frame_file, prev_frame_file)
                frames = np.load(prev_frame_file, mmap_mode="r")
                forward = np.load(fwd_flow_file, mmap_mode="r")
                backward = np.load(bkwd_flow_file, mmap_mode="r")
                consistency = np.load(consistency_file, mmap_mode="r")

            N = frames.shape[0]

            with NpyFile(next_frame_file) as styled:
                frame_range = np.arange(N) if direction > 0 else np.flip(np.arange(N))
                roll = np.random.randint(1, N)
                frame_range = np.roll(frame_range, roll)
                start_idx = frame_range[0] * turbo
                for f_i, f_n in enumerate(frame_range):
                    c_n = (start_idx + f_n * t_d) % len(content)
                    init_img = load_from_memmap(frames, f_n, H, W)

                    if blend > 0:
                        prev_img = load_from_memmap(frames, (f_n - direction) % N, H, W) if f_i == 0 else out_img
                        flow = flow_warp_map((forward if direction == 1 else backward)[f_n], (H, W))
                        flow_mask = load_from_memmap(consistency, f_n, H, W)
                        init_img = flow_weighted(init_img, prev_img, flow, flow_mask, blend, consistency_trust)

                    noise = torch.randn_like(init_img)
                    out_img = diffusion.sample(
                        init_img,
                        prompts=[
                            TextPrompt(text),
                            ContentPrompt(content[c_n].permute(2, 0, 1).div(127.5).sub(1)[None].to(init_img)),
                        ],
                        start_step=n_steps - step + 1,
                        n_steps=blend_every + 1,  # always take an extra step to offset blurring from blending
                        verbose=False,
                        noise=noise,
                    )

                    schedule_step = n_steps - step - 1
                    next_schedule_step = schedule_step - blend_every
                    if next_schedule_step >= 0:
                        sigma = diffusion.diffusion.sqrt_one_minus_alphas_cumprod[next_schedule_step]
                        alpha = diffusion.diffusion.sqrt_alphas_cumprod[next_schedule_step]
                    else:
                        sigma, alpha = 0, 1

                    out_img -= sigma * noise
                    out_img /= alpha

                    styled.append(out_img.cpu().contiguous().numpy())

                    progress.update(blend_every)

            write_video(
                np.load(next_frame_file, mmap_mode="r") * 0.5 + 0.5,
                f"output/{out_name}_{step + 1}.mp4",
                fps=fps / turbo,
            )
            shutil.move(next_frame_file, prev_frame_file)
            direction = -direction  # reverse direction of flow weighting

    write_video(np.load(prev_frame_file, mmap_mode="r") * 0.5 + 0.5, f"output/{out_name}.mp4", fps=fps)
