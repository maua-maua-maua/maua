import os
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from medpy.filter.noise import immerkaer
from npy_append_array import NpyAppendArray as NpyFile
from resize_right import resize
from torch.nn.functional import grid_sample
from tqdm import tqdm

from ..audiovisual.audioreactive.signal import gaussian_filter
from ..flow import get_flow_model
from ..ops.image import immerkaer, local_std, resample
from ..ops.loss import range_loss, tv_loss
from ..ops.video import write_video
from ..style.video import flow_warp_map, preprocess_optical_flow
from .conditioning import CLIPGrads, ContentPrompt, LossGrads, LPIPSGrads, TextPrompt
from .sample import round64
from .wrappers.guided import GuidedDiffusion


def load_from_memmap(mmap, n):
    tensor = torch.from_numpy(mmap[[n]].copy()).cuda()
    if tensor.shape[2] != H or tensor.shape[3] != W:
        tensor = resize(tensor, out_shape=(H, W))
    return tensor.mul(2).sub(1)


def denoise(image, noise, iters=40, mode="local_std"):
    if mode == "local_std":
        total_sigma = local_std(image).mean()
        denoised = image - total_sigma * noise
    elif mode == "immerkaer":
        denoised = image.clone()
        total_sigma = torch.zeros(image.shape[0]).to(image)
        prev_sig = 1000
        for _ in range(iters):
            sigma = immerkaer(denoised)
            if (sigma > prev_sig).any():
                break
            prev_sig = sigma.clone()
            total_sigma += sigma
            if (total_sigma > 1).any():
                sigma -= total_sigma - 1
                total_sigma = total_sigma.clamp(max=1)
                prev_sig = 0  # break on next loop
            denoised = denoised - sigma[:, None, None, None] * noise
    return denoised, total_sigma


if __name__ == "__main__":
    with torch.no_grad():
        W, H = 512, 512
        timesteps = 31
        skip = 1 / 31
        text = "electro-mechanical crystal butterfly wings made of circuits, geometric pattern in the style of cyberpunk noir"
        init = "/home/hans/datasets/video/pleated.mp4"
        blend = 1
        consistency_trust = 0.5
        n_passes = 15
        noise_smooth = 1

        W, H = round64(W), round64(H)
        n_step_total = round((1 - skip) * timesteps)
        n_steps = np.linspace(-1, 0, n_passes) ** 0
        n_steps /= n_steps.sum()
        n_steps *= n_step_total
        n_steps = n_steps.clip(min=1, max=n_step_total).round().astype(int)
        start_steps = np.flip(np.cumsum(np.flip(n_steps)))

        out_name = f"{text.replace(' ','_')}_{str(uuid4())[:6]}"
        if init is not None:
            out_name = f"{Path(init).stem}_{out_name}"

        prev_frame_file = f"workspace/{out_name}_frames_prev.npy"
        next_frame_file = f"workspace/{out_name}_frames_next.npy"
        shutil.rmtree(prev_frame_file, ignore_errors=True)
        shutil.rmtree(next_frame_file, ignore_errors=True)

        content, forward, backward, reliable = preprocess_optical_flow(init, get_flow_model(), smooth=2)
        N = content.shape[0]

        diffusion = GuidedDiffusion(
            [
                CLIPGrads(scale=6500),
                LPIPSGrads(scale=5000),
                LossGrads(tv_loss, scale=60),
                LossGrads(range_loss, scale=75),
            ],
            sampler="plms",
            timesteps=timesteps,
            # model_checkpoint="uncondImageNet256",
        )

        noise = gaussian_filter(torch.randn((N, 3, H, W), device="cuda"), noise_smooth)
        noise /= noise.square().mean().sqrt()

        d = 1
        with tqdm(total=n_step_total * N) as progress:
            for p_n, (start_step, n_steps) in enumerate(zip(start_steps, n_steps)):
                progress.set_description(f"Pass {p_n + 1} / {n_passes}...")

                if os.path.exists(prev_frame_file):
                    frames = np.load(prev_frame_file, mmap_mode="r")
                else:
                    frames = content

                with NpyFile(next_frame_file) as styled:
                    frame_range = list(range(N))
                    start_idx = np.random.randint(0, N)
                    frame_range = frame_range[start_idx:] + frame_range[:start_idx]
                    for f_n in frame_range:

                        curr_img = load_from_memmap(frames, f_n)
                        prev_img = load_from_memmap(frames, (f_n - d) % N)
                        init_img = curr_img.clone()

                        if blend > 0 and p_n > 0:
                            init_clean, sigma = denoise(init_img, noise[[f_n]])
                            prev_clean, _ = denoise(prev_img, noise[[(f_n - d) % N]])

                            flow_map = flow_warp_map((forward if d == 1 else backward)[f_n], (H, W)).cuda()
                            prev_warp = grid_sample(
                                prev_clean, flow_map, padding_mode="reflection", align_corners=False
                            )
                            flow_mask = resample(torch.from_numpy(reliable[None, [f_n]].copy()).cuda(), (H, W))
                            flow_mask = (1 - consistency_trust) + flow_mask * consistency_trust
                            flow_mask *= blend
                            init_clean += flow_mask * prev_warp
                            init_clean /= 1 + flow_mask

                            init_img = init_clean + sigma * noise[[f_n]]

                        out_img = diffusion.sample(
                            init_img,
                            prompts=[TextPrompt(text), ContentPrompt(load_from_memmap(content, f_n))],
                            start_step=start_step,
                            n_steps=n_steps,
                            verbose=False,
                            q_sample=start_step if p_n == 0 else 0,  # start_step // (timesteps // 2),
                            noise=noise[[f_n]],
                        )

                        styled.append(out_img.add(1).div(2).cpu().contiguous().numpy())

                        progress.update(n_steps)

                # TODO more memory efficient to load incrementally with VideoWriter instead of write_video
                new_frames = torch.from_numpy(np.load(next_frame_file)).mul(2).sub(1)
                noise = noise[frame_range]  # rotate noise to match frames
                denoised_frames = denoise(new_frames, noise.cpu())[0]
                write_video(denoised_frames.add(1).div(2), f"output/{out_name}_{p_n}.mp4", fps=12)

                shutil.move(next_frame_file, prev_frame_file)
                d = -d  # reverse direction of flow weighting

        write_video(np.clip(np.load(prev_frame_file), 0, 1), f"output/{out_name}.mp4", fps=12)
