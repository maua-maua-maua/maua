import argparse
import os
from typing import Optional
import warnings
from pathlib import Path

import torch
from diffusers import ControlNetModel, DPMSolverMultistepScheduler, StableDiffusionControlNetImg2ImgPipeline
from torch import Tensor
from torch.nn.functional import grid_sample
from torchvision.io.video import read_video, write_video
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.transforms.functional import resize
from torchvision.utils import flow_to_image
from tqdm import trange
import urllib.request

from tqdm import tqdm

raft_transform = Raft_Large_Weights.DEFAULT.transforms()


def download(url: str, output_path: str) -> None:
    class DownloadProgressBar(tqdm):
        def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


class FILM(torch.jit.ScriptModule):
    def __init__(self, precision=16):
        super().__init__()
        url = f"https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.0/film_net_fp{precision}.pt"
        cache_path = os.path.expanduser(
            os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
        )
        file_path = os.path.join(cache_path, f"film_net_fp{precision}.pt")
        if not os.path.exists(file_path):
            download(url, file_path)
        self.model = torch.jit.load(file_path).eval()
        self.precision = precision

    def forward(self, frame1: Tensor, frame2: Tensor, dt: Tensor) -> Tensor:
        dtype = torch.half if self.precision == 16 else torch.float
        return torch.cat([
            self.model(f[None].to(dtype), s[None].to(dtype), d[None, None].to(dtype))
            for f, s, d in zip(frame1, frame2, dt.flatten())
        ]).to(frame1.dtype)


@torch.inference_mode()
def stylize_video(
    input_video: Tensor,
    prompt: str,
    strength: float = 0.7,
    num_steps: int = 20,
    guidance_scale: float = 7.5,
    controlnet_scale: float = 1.0,
    batch_size: int = 4,
    height: int = 512,
    width: int = 512,
    device: str = "cuda",
) -> Tensor:
    """
    Stylize a video with temporal coherence (less flickering!) using HuggingFace's Stable Diffusion ControlNet pipeline.

    Args:
        input_video (Tensor): Input video tensor of shape (T, C, H, W) and range [0, 1].
        prompt (str): Text prompt to condition the diffusion process.
        strength (float, optional): How heavily stylization affects the image.
        num_steps (int, optional): Number of diffusion steps (tradeoff between quality and speed).
        guidance_scale (float, optional): Scale of the text guidance loss (how closely to adhere to text prompt).
        controlnet_scale (float, optional): Scale of the ControlNet conditioning (strength of temporal coherence).
        batch_size (int, optional): Number of frames to diffuse at once (faster but more memory intensive).
        height (int, optional): Height of the output video.
        width (int, optional): Width of the output video.
        device (str, optional): Device to run stylization process on.

    Returns:
        Tensor: Output video tensor of shape (T, C, H, W) and range [0, 1].
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence annoying TypedStorage warnings

        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=ControlNetModel.from_pretrained("wav/TemporalNet2", torch_dtype=torch.float16),
            safety_checker=None,
            torch_dtype=torch.float16,
        ).to(device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe._progress_bar_config = dict(disable=True)

    raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).eval().to(device)

    n_frames = input_video.shape[0]
    input_video = torch.cat((input_video[[-1]], input_video))

    output_video = []
    for i in trange(1, n_frames, batch_size, desc="Diffusing...", unit="frame", unit_scale=batch_size):
        prev = resize(input_video[i - 1 : i - 1 + batch_size], (height, width), antialias=True).to(device)
        curr = resize(input_video[i : i + batch_size], (height, width), antialias=True).to(device)
        prev = prev[: curr.shape[0]]  # make sure prev and curr have the same batch size (for the last batch)

        flow_img = flow_to_image(raft.forward(*raft_transform(prev, curr))[-1]).div(255)
        control_img = torch.cat((prev, flow_img), dim=1)

        output, _ = pipe(
            prompt=[prompt] * curr.shape[0],
            image=curr,
            control_image=control_img,
            height=height,
            width=width,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            output_type="pt",
            return_dict=False,
        )

        output_video.append(output.cpu())

    return torch.cat(output_video)


def flow_to_warp_map(flow):
    h, w = flow.shape[2:]
    warp_map = torch.meshgrid(torch.linspace(-1, 1, w), torch.linspace(-1, 1, h), indexing="xy")
    warp_map = torch.stack(warp_map).unsqueeze(0).expand_as(flow).to(flow).clone()
    warp_map[:, 0] += flow[:, 0].div(flow.shape[3])
    warp_map[:, 1] += flow[:, 1].div(flow.shape[2])
    return warp_map.permute(0, 2, 3, 1)


def flow_warp(img, flow):
    return grid_sample(img, flow, align_corners=True, padding_mode="reflection")


@torch.inference_mode()
def stylize_video_keyframe(
    input_video: Tensor,
    keyframe_interval: int = 16,
    strength: float = 0.7,
    controlnet_scale: float = 1.0,
    batch_size: int = 4,
    height: int = 512,
    width: int = 512,
    device: str = "cuda",
    **kwargs,
):
    input_video = resize(input_video, (height, width), antialias=True)
    n_frames = len(input_video)
    keyframes = input_video[::keyframe_interval]
    n_keyframes = keyframes.shape[0]

    raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).eval().to(device)
    film = FILM().to(device)

    input_video = torch.cat((input_video, input_video[[0]]))

    forward_flows, backward_flows = [], []
    for i in trange(0, n_frames, batch_size, desc="Calculating flow...", unit="frame", unit_scale=batch_size):
        curr = input_video[i : i + batch_size].to(device)
        next = input_video[i + 1 : i + 1 + batch_size].to(device)
        curr = curr[: next.shape[0]]

        curr, next = raft_transform(curr, next)

        # these flows are time-inverted because grid_sample is confusing!
        forward_flow = raft.forward(next, curr)[-1]
        backward_flow = raft.forward(curr, next)[-1]

        forward_flows.append(forward_flow.cpu())
        backward_flows.append(backward_flow.cpu())
    write_video(
        "flow.mp4",
        flow_to_image(torch.cat(forward_flows)).permute(0, 2, 3, 1).mul(255),
        fps=12,
        options={"crf": "17", "pix_fmt": "yuv420p"},
    )
    forward_flows = flow_to_warp_map(torch.cat(forward_flows))
    backward_flows = flow_to_warp_map(torch.cat(backward_flows))

    print("Diffusing keyframes...")
    keyframes = stylize_video(
        input_video=keyframes,
        strength=strength,
        controlnet_scale=controlnet_scale,
        batch_size=batch_size,
        height=height,
        width=width,
        device=device,
        **kwargs,
    )

    warped_video = []
    for i in trange(1, n_keyframes + 1, desc="Warping...", unit="frame", unit_scale=keyframe_interval):
        forward_frame = keyframes[[i - 1]].to(device)
        backward_frame = keyframes[[i % n_keyframes]].to(device)

        n_blend_frames = (n_frames - (n_keyframes - 1) * keyframe_interval) if i == n_keyframes else keyframe_interval

        forwarped, backwarped = [], []
        for ii in range(n_blend_frames):
            f_idx = (i - 1) * keyframe_interval + ii
            b_idx = min(i * keyframe_interval, n_frames) - ii - 1

            forward_flow = forward_flows[[f_idx]].to(forward_frame)
            backward_flow = backward_flows[[b_idx]].to(backward_frame)

            forward_frame = flow_warp(forward_frame, forward_flow)
            backward_frame = flow_warp(backward_frame, backward_flow)

            forwarped.append(forward_frame)
            backwarped.insert(0, backward_frame)
        forwarped, backwarped = torch.cat(forwarped), torch.cat(backwarped)

        blend_weight = torch.linspace(0, 1, n_blend_frames + 1, device=device)[:-1]
        if film is not None:
            blended = film(forwarped, backwarped, blend_weight).clamp(0, 1)
        else:
            blended = (
                forwarped * (1 - blend_weight[:, None, None, None]) + backwarped * blend_weight[:, None, None, None]
            )
        warped_video.append(blended.cpu())
    warped_video = torch.cat(warped_video)

    write_video(
        "warp.mp4", warped_video.permute(0, 2, 3, 1).mul(255), fps=12, options={"crf": "17", "pix_fmt": "yuv420p"}
    )

    # content_weight = 0.25
    # if film is not None:
    #     warped_video = film(warped_video, input_video[:-1], torch.full((1, 1), content_weight))
    # else:
    #     warped_video = (1 - content_weight) * warped_video + content_weight * input_video[:-1]

    output_video = stylize_video(
        warped_video,
        strength=0.1,
        controlnet_scale=1.0,
        batch_size=batch_size,
        height=height,
        width=width,
        device=device,
        **kwargs,
    )
    return output_video


def argument_parser():
    parser = argparse.ArgumentParser(usage=stylize_video.__doc__)
    parser.add_argument("-i", "--in-file", type=str, required=True)
    parser.add_argument("-p", "--prompt", type=str, required=True)
    parser.add_argument("-o", "--out-file", type=str, default=None)
    parser.add_argument("-s", "--strength", type=float, default=1.0)
    parser.add_argument("-S", "--num-steps", type=int, default=20)
    parser.add_argument("-g", "--guidance-scale", type=float, default=7.5)
    parser.add_argument("-c", "--controlnet-scale", type=float, default=1.0)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-H", "--height", type=int, default=512)
    parser.add_argument("-W", "--width", type=int, default=512)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    return parser


def main(args):
    input_video, _, info = read_video(args.in_file, pts_unit="sec", output_format="TCHW")
    input_video = input_video.div(255)

    output_video = stylize_video(
        input_video=input_video,
        prompt=args.prompt,
        strength=args.strength,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        controlnet_scale=args.controlnet_scale,
        height=args.height,
        width=args.width,
        device=args.device,
        batch_size=args.batch_size,
    )

    out_file = f"{Path(args.in_file).stem} | {args.prompt}.mp4" if args.out_file is None else args.out_file
    write_video(
        out_file, output_video.permute(0, 2, 3, 1).mul(255), fps=12, options={"crf": "17", "pix_fmt": "yuv420p"}
    )


if __name__ == "__main__":
    main(argument_parser().parse_args())
