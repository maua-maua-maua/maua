from pathlib import Path

import torch

torch.cuda.is_available()

# torch MUST be imported before decord for reasons?!

import decord

decord.bridge.set_bridge("torch")

import os
import shutil
from functools import partial, reduce
from typing import Callable, Optional, Tuple, Union

import easydict
import matplotlib.pyplot as plt
import numpy as np
from npy_append_array import NpyAppendArray
from torch.nn.functional import grid_sample
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
from tqdm import trange

from ..flow import get_flow_model
from ..flow.lib import flow_warp_map, get_consistency_map
from ..ops.image import match_histogram, sharpen
from ..ops.video import write_video
from ..prompt import ContentPrompt, StylePrompt, TextPrompt
from ..utility import seed_everything
from .image import build_output_name, get_diffusion_model, round64, width_height
from .processors.base import BaseDiffusionProcessor


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
    def __init__(self, file, height, width, device, load=False):
        super().__init__()
        self.file, self.height, self.width = file, height, width
        self.device = device
        self.array = np.load(self.file, mmap_mode="r") if load else None

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


def initialize_cache_files(names, out_name, length, height, width, device):
    caches = {}
    for name in names:
        filename = f"workspace/{out_name}_{name}.npy"
        reuse = False

        if os.path.exists(filename):
            arr = np.load(filename, mmap_mode="r")
            reuse = (
                arr.shape[0] == length
                and arr.shape[2 if name == "consistency" else 1] == height
                and arr.shape[3 if name == "consistency" else 2] == width
            )
            if reuse:
                print(f"{name} cache seems valid, reusing...")
            else:
                os.remove(filename)

        caches[name] = MemoryMappedFrames(filename, height, width, device, load=reuse)

    return easydict.EasyDict(caches)


def initialize_optical_flow(cache, frames):
    if len(cache.flow) > 0 and len(cache.consistency) > 0:
        return

    flow_model = get_flow_model()
    N = len(frames)
    with cache.flow, cache.reverse, cache.consistency:
        for f_n in trange(N, desc="Calculating optical flow..."):
            prev = frames[(f_n - 1) % N].add(1).div(2)
            curr = frames[f_n].add(1).div(2)
            ff = flow_model(curr, prev)
            bf = flow_model(prev, curr)
            cache.flow.append(ff)
            cache.reverse.append(bf)
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
        first_skip: float = 0.4,
        skip: float = 0.7,
        blend: float = 2,
        consistency_trust: float = 0.75,
        wrap_around: int = 0,
        turbo: int = 1,
        flow_exaggeration: float = 1.0,
        pre_hook: Optional[Callable] = None,
        post_hook: Optional[Callable] = None,
        hist_persist: bool = False,
        constant_seed: Optional[int] = None,
        device: str = "cuda",
        preview: bool = False,
    ):
        # process user inputs
        width, height = [round64(s) for s in size]
        n_steps = round((1 - skip) * timesteps)
        first_steps = round((1 - first_skip) * timesteps)

        # load init video
        frames = VideoFrames(init, height, width, device)
        N = len(frames)

        # initialize cache files
        cache = initialize_cache_files(
            names=["out", "flow", "reverse", "consistency"],
            out_name=build_output_name(init, style=None, text=None, unique=False),
            length=N,
            height=height,
            width=width,
            device=device,
        )

        # calculate optical flow
        initialize_optical_flow(cache, frames)

        try:
            fade = torch.sqrt(torch.linspace(1, 0, wrap_around)).reshape(-1, 1, 1, 1).to(device)

            f_start = len(cache.out)  # TODO resume cancelled stylization?
            with cache.out:
                for f_n in trange(f_start, N + wrap_around):

                    if constant_seed:
                        seed_everything(constant_seed)

                    if f_n % turbo != 0:
                        out_img = warp(out_img, flow_warp_map(cache.flow[f_n % N] * flow_exaggeration))
                        cache.out.append(out_img)
                        continue

                    prompts = [ContentPrompt(frames[f_n % N])]
                    if text is not None:
                        prompts.append(TextPrompt(text))
                    if style is not None:
                        prompts.append(StylePrompt(path=style, size=(height, width)))

                    init_img = frames[f_n % N]

                    if blend > 0:
                        flow_mask = cache.consistency[f_n % N]
                        flow_mask *= consistency_trust
                        flow_mask += 1 - consistency_trust
                        flow_mask *= blend

                        flow = flow_warp_map(cache.flow[f_n % N] * flow_exaggeration)

                        prev_img = frames[(f_n - 1) % N] if f_n == 0 else out_img
                        prev_warp = warp(prev_img, flow)

                        init_img += flow_mask * prev_warp
                        init_img /= 1 + flow_mask

                    if f_n / N >= 1:
                        last_round = torch.from_numpy(np.load(cache.out.file, mmap_mode="r")[f_n % N].copy()).to(device)
                        init_img = fade[[f_n % N]] * init_img + (1 - fade[[f_n % N]]) * last_round

                    if pre_hook:
                        init_img = pre_hook(init_img)

                    if hist_persist and f_n > 0:
                        init_img = match_histogram(init_img, hist_img)

                    out_img = diffusion(
                        init_img, prompts=prompts, start_step=first_steps if f_n == 0 else n_steps, verbose=False
                    )

                    if hist_persist and f_n == 0:
                        hist_img = out_img.clone()

                    if post_hook:
                        out_img = post_hook(out_img)

                    if preview:
                        plt.imshow(to_pil_image(out_img.squeeze().add(1).div(2).clamp(0, 1)))
                        plt.axis("off")
                        plt.show(block=False)
                        plt.pause(0.5)

                    cache.out.append(out_img)

        except KeyboardInterrupt:
            return np.load(cache.out.file, mmap_mode="r") * 0.5 + 0.5

        output = np.load(cache.out.file, mmap_mode="r+")
        output *= 0.5
        output += 0.5
        if wrap_around > 0:
            output[:wrap_around] = output[-wrap_around:]
        return output[:N]


@torch.no_grad()
def video_sample(
    diffusion: Union[str, BaseDiffusionProcessor],
    init: str,
    text: Optional[str] = None,
    style: Optional[str] = None,
    size: Tuple[int] = (256, 256),
    timesteps: int = 50,
    first_skip: float = 0.4,
    skip: float = 0.7,
    blend: float = 2,
    consistency_trust: float = 0.75,
    wrap_around: int = 0,
    turbo: int = 1,
    flow_exaggeration: float = 1,
    sampler: str = "plms",
    guidance_speed: str = "fast",
    clip_scale: float = 0.0,
    lpips_scale: float = 0.0,
    style_scale: float = 0.0,
    color_match_scale: float = 0.0,
    cfg_scale: float = 5.0,
    match_hist: bool = False,
    hist_persist: bool = False,
    sharpness: float = 1.0,
    inter_noise: float = 0.0,
    constant_seed: Optional[int] = None,
    device: str = "cuda",
    preview: bool = False,
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

    pre_hook = partial(match_histogram, source_tensor=StylePrompt(path=style).img) if match_hist else None

    post_fns = []
    if sharpness != 1.0:
        post_fns.append(partial(sharpen, strength=sharpness))
    if inter_noise > 0.0:
        post_fns.append(lambda img: img + inter_noise * torch.randn_like(img))
    post_hook = (lambda img: reduce(lambda i, f: f(i), post_fns, img)) if len(post_fns) > 0 else None

    video = VideoFlowDiffusionProcessor()(
        diffusion=diffusion,
        init=init,
        text=text,
        style=style,
        size=size,
        timesteps=timesteps,
        first_skip=first_skip,
        skip=skip,
        blend=blend,
        consistency_trust=consistency_trust,
        wrap_around=wrap_around,
        turbo=turbo,
        flow_exaggeration=flow_exaggeration,
        pre_hook=pre_hook,
        post_hook=post_hook,
        hist_persist=hist_persist,
        constant_seed=constant_seed,
        device=device,
        preview=preview,
    )
    return video


if __name__ == "__main__":
    # fmt:off
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=True)
    parser.add_argument("--init", type=str, default="random", help='How to initialize the image "random", "perlin", or a path to an image file.')
    parser.add_argument("--text", type=str, default=None, help='A text prompt to visualize.')
    parser.add_argument("--style", type=str, default=None, help='An image whose style should be optimized for in the output image (only works with "guided" diffusion at the moment, see --style-scale).')
    parser.add_argument("--size", type=width_height, default=(512, 512), help='Size to synthesize the video at.')
    parser.add_argument("--skip", type=float, default=0.85, help='Lower fractions will stray further from the original image, while higher fractions will hallucinate less detail.')
    parser.add_argument("--first-skip", type=float, default=0.4, help='Separate skip fraction for the first frame.')
    parser.add_argument("--timesteps", type=int, default=50, help='Number of timesteps to sample the diffusion process at. Higher values will take longer but are generally of higher quality.')
    parser.add_argument("--blend", type=float, default=2, help='Factor with which to blend previous frames into the next frame. Higher values will stay more consistent over time (e.g. --blend 20 means 20:1 ratio of warped previous frame to new input frame).')
    parser.add_argument("--consistency-trust", type=float, default=0.75, help='How strongly to trust flow consistency mask. Lower values will lead to more consistency over time. Higher values will respect occlusions of the background more.')
    parser.add_argument("--wrap-around", type=int, default=0, help='Number of extra frames to continue for, looping back to start. This allows for seamless transitions back to the start of the video.')
    parser.add_argument("--turbo", type=int, default=1, help='Only apply diffusion every --turbo\'th frame, otherwise just warp the previous frame with optical flow. Can be much faster for high factors at the cost of some visual detail.')
    parser.add_argument("--flow-exaggeration", type=float, default=1, help='Factor to multiply optical flow with. Higher values lead to more extreme movements in the final video.')
    parser.add_argument("--diffusion", type=str, default="stable", help='Which diffusion model to use. Options: "guided", "latent", "glide", "glid3xl", "stable" or a /path/to/stable-diffusion.ckpt')
    parser.add_argument("--sampler", type=str, default="dpm_2", choices=["p", "ddim", "plms", "euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms"], help='Which sampling method to use. "p", "ddim", and "plms" work for all diffusion models, the rest are currently only supported with "stable" diffusion.')
    parser.add_argument("--guidance-speed", type=str, default="fast", choices=["regular", "fast"], help='How to perform "guided" diffusion. "regular" is slower but can be higher quality, "fast" corresponds to the secondary model method (a.k.a. Disco Diffusion).')
    parser.add_argument("--clip-scale", type=float, default=0.0, help='Controls strength of CLIP guidance when using "guided" diffusion.')
    parser.add_argument("--lpips-scale", type=float, default=0.0, help='Controls the apparent influence of the content image when using "guided" diffusion and a --content image.')
    parser.add_argument("--style-scale", type=float, default=0.0, help='When using "guided" diffusion and a --style image, a higher --style-scale enforces textural similarity to the style, while a lower value will be conceptually similar to the style.')
    parser.add_argument("--color-match-scale", type=float, default=0.0, help='When using "guided" diffusion, the --color-match-scale guides the output\'s colors to match the --style image.')
    parser.add_argument("--cfg-scale", type=float, default=7.5, help='Classifier-free guidance strength. Higher values will match the text prompt more closely at the cost of output variability.')
    parser.add_argument("--match-hist", action="store_true", help='Match the color histogram of the initialization image to the --style image before starting diffusion.')
    parser.add_argument("--hist-persist", action="store_true", help='Match the color histogram of subsequent frames to the first diffused frame (helps alleviate oversaturation).')
    parser.add_argument("--sharpness", type=float, default=1.0, help='Sharpen the image by this amount after each diffusion scale (a value of 1.0 will leave the image unchanged, higher values will be sharper).')
    parser.add_argument("--inter-noise", type=float, default=0.0, help='Scale of extra noise added between frames (helps reduce empty spaces forming in the video).')
    parser.add_argument("--constant-seed", type=int, default=None, help='Use a fixed noise seed for all frames (None to disable).')
    parser.add_argument("--device", type=str, default="cuda", help='Which device to use (e.g. "cpu" or "cuda:1")')
    parser.add_argument("--preview", action="store_true", help='Show frames as they\'re rendered (moderately slower).')
    parser.add_argument("--fps", type=int, default=12, help='Framerate of output video.')
    parser.add_argument("--out-dir", type=str, default="output/", help='Directory to save output images to.')
    args = parser.parse_args()
    # fmt:on

    out_name = build_output_name(args.init, args.style, args.text)[:222]
    out_dir, fps = args.out_dir, args.fps
    del args.out_dir, args.fps

    video = video_sample(**vars(args))

    write_video(video, f"output/{Path(args.diffusion).stem}_{out_name}.mp4", fps=fps)
