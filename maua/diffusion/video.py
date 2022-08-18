import torch

torch.cuda.is_available()

# torch MUST be imported before decord for reasons?!

import decord

decord.bridge.set_bridge("torch")

import os
import shutil
from typing import Optional, Tuple, Union

import easydict
import numpy as np
from npy_append_array import NpyAppendArray
from torch.nn.functional import grid_sample
from torch.utils.data import Dataset
from tqdm import trange

from ..flow import get_flow_model
from ..flow.lib import flow_warp_map, get_consistency_map
from ..ops.video import write_video
from ..prompt import ContentPrompt, StylePrompt, TextPrompt
from .multires import round64
from .processors.base import BaseDiffusionProcessor
from .sample import build_output_name, get_diffusion_model, width_height


class VideoFrames(Dataset):
    def __init__(self, filename, height, width, device):
        super().__init__()
        self.reader = decord.VideoReader(filename, width=width, height=height)
        self.prepare = lambda x: x.permute(2, 0, 1).unsqueeze(0).div(127.5).sub(1).to(device)

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray, torch.Tensor)):
            return torch.stack([self.prepare(self.reader[i]) for i in idx])
        return self.prepare(self.reader[idx])


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


def initialize_cache_files(names, out_name, height, width, device):
    caches = {}
    for cache_file in names:
        filename = f"workspace/{out_name}_{cache_file}.npy"
        if os.path.exists(filename):
            os.remove(filename)
        caches[cache_file] = MemoryMappedFrames(filename, height, width, device)
    return easydict.EasyDict(caches)


def initialize_optical_flow(cache, frames):
    flow_model = get_flow_model()
    N = len(frames)
    with cache.flow, cache.consistency:
        for f_n in trange(N, desc="Calculating optical flow..."):
            prev = frames[(f_n - 1) % N].add(1).div(2)
            curr = frames[f_n].add(1).div(2)
            ff = flow_model(curr, prev)
            bf = flow_model(prev, curr)
            cache.flow.append(flow_warp_map(ff))
            cache.consistency.append(get_consistency_map(ff, bf))


def warp(x, f):
    return grid_sample(x, f, padding_mode="reflection", align_corners=False)


class VideoFlowDiffusionProcessor(torch.nn.Module):
    def forward(
        self,
        diffusion: BaseDiffusionProcessor,
        init: str,
        text: str = None,
        style: str = None,
        size: Tuple[int] = (256, 256),
        timesteps: int = 100,
        skip: float = 0.4,
        blend: float = 2,
        consistency_trust: float = 0.75,
        wrap_around: int = 0,
        turbo: int = 1,
        device: str = "cuda",
    ):
        # process user inputs
        width, height = [round64(s) for s in size]
        n_steps = round((1 - skip) * timesteps)

        # load init video
        frames = VideoFrames(init, height, width, device)
        N = len(frames)

        # initialize cache files
        cache = initialize_cache_files(
            ["out", "flow", "consistency"], build_output_name(init, style, text), height, width, device
        )

        # calculate optical flow
        initialize_optical_flow(cache, frames)

        with torch.no_grad(), cache.out:
            for f_n in trange(N):

                if f_n % turbo != 0:
                    out_img = warp(out_img, cache.flow[f_n % N])
                    cache.out.append(out_img)
                    continue

                init_img = frames[f_n % N]

                if blend > 0:
                    flow_mask = cache.consistency[f_n % N]
                    flow_mask *= consistency_trust
                    flow_mask += 1 - consistency_trust
                    flow_mask *= blend

                    flow = cache.flow[f_n % N]

                    prev_img = frames[(f_n - 1) % N] if f_n == 0 else out_img

                    init_img += flow_mask * warp(prev_img, flow)
                    init_img /= 1 + flow_mask

                prompts = [ContentPrompt(frames[f_n % N])]
                if text is not None:
                    prompts.append(TextPrompt(text))
                if style is not None:
                    prompts.append(StylePrompt(path=style, size=(height, width)))

                out_img = diffusion(init_img, prompts=prompts, start_step=n_steps, verbose=False)

                cache.out.append(out_img)

        output = np.load(cache.out.file, mmap_mode="r+") * 0.5 + 0.5
        if wrap_around > 0:
            fade = np.linspace(0, 1, wrap_around).reshape(-1, 1, 1, 1)
            output[:wrap_around] = (1 - fade) * output[:wrap_around] + fade * output[-wrap_around:]

        return output[:N]


def main(
    diffusion: Union[str, BaseDiffusionProcessor],
    init: str,
    text: Optional[str] = None,
    style: Optional[str] = None,
    size: Tuple[int] = (256, 256),
    timesteps: int = 50,
    skip: float = 0.4,
    blend: float = 2,
    consistency_trust: float = 0.75,
    wrap_around: int = 0,
    turbo: int = 1,
    sampler: str = "plms",
    guidance_speed: str = "fast",
    clip_scale: float = 2500.0,
    lpips_scale: float = 0.0,
    style_scale: float = 0.0,
    color_match_scale: float = 0.0,
    cfg_scale: float = 5.0,
    device: str = "cuda",
):
    diffusion = get_diffusion_model(
        diffusion=diffusion,
        timesteps=timesteps,
        sampler=sampler,
        guidance_speed=guidance_speed,
        clip_scale=clip_scale,
        lpips_scale=lpips_scale,
        style_scale=style_scale,
        color_match_scale=color_match_scale,
        cfg_scale=cfg_scale,
    )
    video = VideoFlowDiffusionProcessor()(
        diffusion=diffusion,
        init=init,
        text=text,
        style=style,
        size=size,
        timesteps=timesteps,
        skip=skip,
        blend=blend,
        consistency_trust=consistency_trust,
        wrap_around=wrap_around,
        turbo=turbo,
        device=device,
    )
    return video


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=True)
    parser.add_argument("--init", type=str)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--style", type=str, default=None)
    parser.add_argument("--size", type=width_height, default=(256, 256))
    parser.add_argument("--skip", type=float, default=0.4)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--blend", type=float, default=2)
    parser.add_argument("--consistency_trust", type=float, default=0.75)
    parser.add_argument("--wrap_around", type=int, default=0)
    parser.add_argument("--turbo", type=int, default=1)
    parser.add_argument("--diffusion", type=str, default="guided", choices=["guided", "latent", "glide", "glid3xl"])
    parser.add_argument("--sampler", type=str, default="plms", choices=["p", "ddim", "plms"])
    parser.add_argument("--guidance-speed", type=str, default="fast", choices=["regular", "fast", "hyper"])
    parser.add_argument("--clip-scale", type=float, default=2500.0)
    parser.add_argument("--lpips-scale", type=float, default=0.0)
    parser.add_argument("--style-scale", type=float, default=0.0)
    parser.add_argument("--color-match-scale", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--out-dir", type=str, default="output/")
    args = parser.parse_args()

    out_name = build_output_name(args.init, args.style, args.text)[:222]
    out_dir, fps = args.out_dir, args.fps
    del args.out_dir, args.fps

    video = main(**vars(args))

    write_video(video, f"output/{out_name}.mp4", fps=fps)
