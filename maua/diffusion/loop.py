import os
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from kornia.filters.unsharp import unsharp_mask
from npy_append_array import NpyAppendArray as NpyFile
from resize_right import resize
from torch.nn.functional import grid_sample
from tqdm import tqdm

from ..audiovisual.audioreactive.signal import gaussian_filter
from ..flow import get_flow_model
from ..ops.loss import range_loss, tv_loss
from ..ops.video import write_video
from ..style.video import flow_warp_map, preprocess_optical_flow
from .conditioning import CLIPGrads, ContentPrompt, LossGrads, LPIPSGrads, TextPrompt
from .sample import round64
from .wrappers.guided import GuidedDiffusion


def load_from_memmap(mmap, n):
    tensor = torch.from_numpy(mmap[[n]].copy()).cuda()
    if tensor.dim() < 4:
        tensor = tensor.unsqueeze(1)
    if tensor.shape[2] != H or tensor.shape[3] != W:
        tensor = resize(tensor, out_shape=(H, W))
    return tensor.mul(2).sub(1)


if __name__ == "__main__":
    with torch.no_grad():
        W, H = 512, 512
        timesteps = 25
        skip = 10 / 25
        text = "an epic illustration of a futuristic city made of magical runes and mathematical symbols, grayscale pen ink illustration"
        init = "/home/hans/modelzoo/diffusionGAN/denoising/take0/meer netsj interps/_diffusionGAN_interpolation_denoising_epoch82_seed48865_5065ae.mp4"
        blend = 1
        consistency_trust = 1
        noise_smooth = 1
        blend_every = 1
        fps = 12
        continue_previous = False  # not working yet

        W, H = round64(W), round64(H)
        n_steps = round((1 - skip) * timesteps)
        out_name = f"{text.replace(' ','_')}"
        if init is not None:
            out_name = f"{Path(init).stem}_{out_name}"
        prev_frame_file = f"workspace/{out_name}_frames_prev.npy"
        next_frame_file = f"workspace/{out_name}_frames_next.npy"
        out_name += f"_{str(uuid4())[:6]}"

        if not continue_previous:
            shutil.rmtree(prev_frame_file, ignore_errors=True)
        shutil.rmtree(next_frame_file, ignore_errors=True)

        content, forward, backward, reliable = preprocess_optical_flow(init, get_flow_model(), smooth=2)
        N = content.shape[0]

        diffusion = GuidedDiffusion(
            [
                CLIPGrads(scale=6500),
                LPIPSGrads(scale=1000),
                LossGrads(tv_loss, scale=60),
                LossGrads(range_loss, scale=75),
            ],
            sampler="plms",
            timesteps=timesteps,
        )

        # TODO replace with lower memory version
        noise = gaussian_filter(torch.randn((N, 3, H, W), device="cuda"), noise_smooth)
        noise /= noise.square().mean().sqrt()

        d = 1
        with tqdm(total=n_steps * N) as progress:
            for step in range(0, n_steps, blend_every):
                progress.set_description(f"Step {step + 1} / {n_steps}...")

                if os.path.exists(prev_frame_file):
                    frames = np.load(prev_frame_file, mmap_mode="r")
                else:
                    frames = content

                schedule_step = n_steps - step - 1
                sigma = diffusion.diffusion.sqrt_one_minus_alphas_cumprod[schedule_step]
                alpha = diffusion.diffusion.sqrt_alphas_cumprod[schedule_step]

                next_schedule_step = schedule_step - blend_every
                if next_schedule_step >= 0:
                    next_sigma = diffusion.diffusion.sqrt_one_minus_alphas_cumprod[next_schedule_step]
                    next_alpha = diffusion.diffusion.sqrt_alphas_cumprod[next_schedule_step]
                else:
                    next_sigma, next_alpha = 0, 1

                with NpyFile(next_frame_file) as styled:
                    frame_range = np.arange(N) if d > 0 else np.flip(np.arange(N))
                    frame_range = np.roll(frame_range, np.random.randint(0, N))
                    for f_i, f_n in enumerate(frame_range):
                        init_img = load_from_memmap(frames, f_n)

                        if blend > 0:
                            if step > 0:
                                init_img -= sigma * noise[[f_n]]
                                init_img /= alpha

                            prev_img = load_from_memmap(frames, (f_n - d) % N) if f_i == 0 else out_img
                            if not (step == 0 and f_i == 0):
                                prev_img -= (sigma if f_i == 0 else next_sigma) * noise[(f_n - d) % N]
                                prev_img /= alpha if f_i == 0 else next_alpha

                            flow_map = flow_warp_map((forward if d == 1 else backward)[f_n], (H, W)).cuda()
                            prev_warp = grid_sample(prev_img, flow_map, padding_mode="reflection", align_corners=False)
                            flow_mask = load_from_memmap(reliable, f_n).add(1).div(2)
                            flow_mask *= consistency_trust
                            flow_mask += 1 - consistency_trust
                            flow_mask *= blend
                            init_img += flow_mask * prev_warp
                            init_img /= 1 + flow_mask

                        out_img = diffusion.sample(
                            init_img,
                            prompts=[TextPrompt(text), ContentPrompt(load_from_memmap(content, f_n))],
                            start_step=n_steps - step,
                            n_steps=blend_every,
                            verbose=False,
                            noise=noise[[f_n]],
                        )

                        styled.append(out_img.add(1).div(2).cpu().contiguous().numpy())

                        progress.update(blend_every)

                noise = noise[frame_range]  # rotate noise to match frames
                d = -d  # reverse direction of flow weighting

                # TODO more memory efficient to load incrementally with VideoWriter instead of write_video
                new_frames = torch.from_numpy(np.load(next_frame_file)).mul(2).sub(1)
                denoised_frames = new_frames - next_sigma * noise.cpu()
                denoised_frames = denoised_frames.div(next_alpha).add(1).div(2)
                write_video(denoised_frames, f"output/{out_name}_{step + 1}.mp4", fps=fps)

                shutil.move(next_frame_file, prev_frame_file)

        write_video(np.clip(np.load(prev_frame_file), 0, 1), f"output/{out_name}.mp4", fps=fps)
