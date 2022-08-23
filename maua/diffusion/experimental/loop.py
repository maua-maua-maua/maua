import os
import shutil

import decord
import easydict
import numpy as np
import torch
from npy_append_array import NpyAppendArray
from torch.nn.functional import grid_sample
from torch.utils.data import Dataset
from tqdm import tqdm, trange

from ...flow import get_flow_model
from ...flow.lib import flow_warp_map, get_consistency_map
from ...grad import CLIPGrads, ColorMatchGrads, ContentPrompt, LPIPSGrads, StylePrompt, TextPrompt, VGGGrads
from ...ops.video import write_video
from ...super.video.framerate import rife
from ..processors.guided import GuidedDiffusion
from ..image import build_output_name
from ..multires import round64

decord.bridge.set_bridge("torch")


def warp(x, f):
    return grid_sample(x, f, padding_mode="reflection", align_corners=False)


def initialize_cache_files(out_name, height, width, device):
    caches = {}
    for cache_file in ["new", "old", "forward", "backward", "reliable"]:
        filename = f"workspace/{out_name}_{cache_file}.npy"
        if os.path.exists(filename):
            os.remove(filename)
        caches[cache_file] = MemoryMappedFrames(filename, height, width, device)
    return easydict.EasyDict(caches)


class VideoFrames(Dataset):
    def __init__(self, filename, height, width, device):
        super().__init__()
        self.reader = decord.VideoReader(filename, width=width, height=height)
        self.device = device

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray, torch.Tensor)):
            return torch.stack(
                [self.reader[i].permute(2, 0, 1).unsqueeze(0).div(127.5).sub(1).to(self.device) for i in idx]
            )
        return self.reader[idx].permute(2, 0, 1).unsqueeze(0).div(127.5).sub(1).to(self.device)


class MemoryMappedFrames(Dataset):
    def __init__(self, file, height, width, device):
        super().__init__()
        self.file, self.height, self.width = file, height, width
        self.device = device
        self.array = None

    def clear(self):
        os.remove(self.file)
        self.array = None

    def update(self, other):
        shutil.move(other.file, self.file)
        self.array = np.load(self.file, mmap_mode="r")
        other.array = None

    def __len__(self):
        return len(self.array) if self.array is not None else 0

    def __getitem__(self, idx):
        if self.array is None:
            raise Exception("Cache is empty!")
        if not isinstance(idx, (list, np.ndarray)):
            idx = [idx]
        tensor = torch.from_numpy(self.array[idx].copy()).float().to(self.device)
        if tensor.dim() < 4:
            tensor = tensor.unsqueeze(1)
        return tensor

    def __enter__(self):
        self.array = NpyAppendArray(self.file)

    def append(self, item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        if len(item.shape) < 4:
            item = item[None]
        self.array.append(np.ascontiguousarray(item))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.array.close()
        if exc_type:
            return
        self.array = np.load(self.file, mmap_mode="r")


def initialize_optical_flow(cache, frames):
    flow_model = get_flow_model()
    N = len(frames)
    with cache.forward, cache.backward, cache.reliable:
        for f_n in trange(N, desc="Calculating optical flow..."):
            prev = frames[(f_n - 1) % N].add(1).div(2)
            curr = frames[f_n].add(1).div(2)
            ff = flow_model(curr, prev)
            bf = flow_model(prev, curr)
            cache.forward.append(flow_warp_map(ff))
            cache.backward.append(flow_warp_map(bf))
            cache.reliable.append(get_consistency_map(ff, bf))


def update_optical_flow(cache, frames, content, turbo):
    flow_model = get_flow_model()
    interp_model = rife.load_model()

    # remove old cache, initialize new ones
    cache.forward.clear(), cache.backward.clear(), cache.reliable.clear()
    with cache.new, cache.forward, cache.backward, cache.reliable:

        # load a new frame halfway between each pair of
        for f_n in range(len(frames)):
            prev = content[(start_idx + (f_n - 1) * turbo) % len(content)].add(1).div(2)
            btwn = content[(start_idx + round((f_n - 0.5) * turbo)) % len(content)].add(1).div(2)
            curr = content[(start_idx + f_n * turbo) % len(content)].add(1).div(2)

            ff1, ff2 = flow_warp_map(flow_model(btwn, prev)), flow_warp_map(flow_model(curr, btwn))
            bf1, bf2 = flow_warp_map(flow_model(prev, btwn)), flow_warp_map(flow_model(btwn, curr))
            fc1, fc2 = get_consistency_map(ff1, bf1)[None], get_consistency_map(ff2, bf2)[None]

            prev = frames[(f_n - 1) % N]
            curr = frames[f_n]
            prev_warp = warp(prev, ff1)
            curr_warp = warp(curr, bf2)
            if interp_model:
                new = (
                    next(rife.interpolate(prev_warp.add(1).div(2), curr_warp.add(1).div(2), interp_model, factor=1))
                    .float()
                    .mul(2)
                    .sub(1)
                )
            else:
                new = 0.5 * (prev_warp + curr_warp)

            cache.new.append(torch.cat((new, curr)))
            cache.forward.append(torch.cat((ff1, ff2)))
            cache.backward.append(torch.cat((bf1, bf2)))
            cache.reliable.append(torch.cat((fc1, fc2)))

    cache.old.update(cache.new)


if __name__ == "__main__":
    W, H = 256, 256
    timesteps = 40
    skip = 0.6
    blend_every = 2
    blend_first = 3
    blend = 2
    consistency_trust = 0.75
    text = "a very very very beautiful watercolor painting of an epic galaxy filled with pulsars and gas giants"
    init = "/home/hans/datasets/video/dreams.mp4"
    style_img = None
    fps = 12
    clip_scale = 3500
    lpips_scale = 2000
    style_scale = 0
    color_match_scale = 0
    diffusion_speed = "fast"
    diffusion_sampler = "p"
    turbo_start = 1
    diffusion_model = "uncondImageNet256"
    device = "cuda"

    # process user inputs
    W, H = round64(W), round64(H)
    n_steps = round((1 - skip) * timesteps)
    blend_every = round(blend_every * timesteps) if blend_every < 1 else blend_every
    turbo_schedule = [int(turbo_start / 2**i) for i in range(round(np.log2(turbo_start)) + 1)]
    turbo_schedule += [1 for _ in range(len(list(range(0, n_steps, blend_every))) - len(turbo_schedule))]

    # build output name based on inputs
    out_name = build_output_name(init, style_img, text)

    # initialize cache files
    cache = initialize_cache_files(out_name, H, W, device)

    # load init video
    content = VideoFrames(init, H, W, device)

    # initialize diffuser
    diffusion = GuidedDiffusion(
        [
            CLIPGrads(scale=clip_scale),
            LPIPSGrads(scale=lpips_scale),
            VGGGrads(scale=style_scale),
            ColorMatchGrads(scale=color_match_scale),
        ],
        sampler=diffusion_sampler,
        timesteps=timesteps,
        model_checkpoint=diffusion_model,
        speed=diffusion_speed,
    )

    start_idx, direction = 0, 1
    total_steps = sum([round(blend_every * len(content) / turbo) for turbo in turbo_schedule])
    with tqdm(total=total_steps) as progress, torch.no_grad():
        for s_i, (step, turbo) in enumerate(zip(range(0, n_steps, blend_every), turbo_schedule)):
            progress.set_description(f"Step {step + 1} - {step + blend_every} of {n_steps}, Turbo {turbo}...")

            t_d = turbo * direction

            if step == 0:
                # in the first step we initialize with the content video
                idxs = np.arange(0, len(content), turbo)
                frames = content[idxs]  # TODO needs a more memory efficient way

                # we also need to calculate some initial optical flows
                initialize_optical_flow(cache, frames)
            else:
                # load init images for this pass
                frames = cache.old

            # when the turbo schedule changes we double the temporal resolution + recalculate optical flow
            if len(content) / turbo > len(frames):
                update_optical_flow(cache, frames, content, turbo)

            N = len(frames)

            with cache.new:
                frame_range = np.arange(N) if direction > 0 else np.flip(np.arange(N))
                roll = np.random.randint(1, N)
                frame_range = np.roll(frame_range, roll)
                start_idx = frame_range[0] * turbo
                for f_i, f_n in enumerate(frame_range):
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

                    prompts = [ContentPrompt(content[(start_idx + f_n * turbo * direction) % len(content)])]
                    if text is not None:
                        prompts.append(TextPrompt(text))
                    if style_img is not None:
                        prompts.append(StylePrompt(path=style_img, size=(H, W)))

                    out_img = diffusion.sample(
                        init_img,
                        prompts=prompts,
                        start_step=n_steps - step,
                        n_steps=blend_every if s_i < blend_first else None,
                        verbose=False,
                    )
                    cache.new.append(out_img)

                    progress.update(blend_every if s_i < blend_first else n_steps - blend_first * blend_every)

            write_video(
                np.load(cache.new.file, mmap_mode="r") * 0.5 + 0.5, f"output/{out_name}_{step + 1}.mp4", fps=fps / turbo
            )
            cache.old.update(cache.new)
            direction = -direction  # reverse direction of flow weighting

            if s_i >= blend_first:
                break

    write_video(np.load(cache.old.file, mmap_mode="r") * 0.5 + 0.5, f"output/{out_name}.mp4", fps=fps)
