import argparse
from pathlib import Path
from uuid import uuid4

import torch
from tqdm import tqdm

from ..audiovisual.audioreactive.selfsupervised.features.processing import gaussian_filter
from ..ops.video import VideoWriter
from .wrappers import MauaGenerator, get_generator_class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def random_interpolation(G: MauaGenerator, n_frames: int, smooth: float, truncation: torch.Tensor, batch_size: int):
    latents = torch.randn((n_frames, 512), device=device)
    latents = gaussian_filter(latents, sigma=smooth)
    latents /= latents.square().mean().sqrt()
    for latent_batch in tqdm(
        latents.split(batch_size), unit="frames", unit_scale=batch_size, desc="Rendering interpolation..."
    ):
        for frame in G.forward(latent_batch, truncation=truncation):
            yield frame


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", required=True, type=str, help="Path to .pkl/.pt file containing the model to use")
    parser.add_argument("--architecture", default="stylegan2", type=str, choices=["stylegan2", "stylegan3"], help="The architecture of the model")
    parser.add_argument("--n_frames", default=300, type=int, help="Number of frames to render in the output video")
    parser.add_argument("--fps", default=30, type=float, help="Framerate for the output video")
    parser.add_argument("--smooth", default=5, type=float, help="How much to smooth the latents by. Will influence how fast the video seems to interpolate. Higher values give slower interpolations.")
    parser.add_argument("--truncation", default=1.0, type=float, help="Latent truncation value. Lower values give higher quality with less diversity; higer values vice versa")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--out_size", default="1024,1024", type=str, help="Desired width,height of output image: e.g. 1920,1080 or 720,1280")
    parser.add_argument("--resize_strategy", default="stretch", type=str, help="Strategy used to resize (in feature space) to achieve desired output resolution")
    parser.add_argument("--resize_layer", default=0, choices=list(range(18)), type=int, help="Which layer in the network to perform resizing at. Higher values are closer to resizing output pixels directly. Lower values have larger rounding increments (i.e. less flexible possible output sizes)")
    parser.add_argument("--out_dir", default="./output/", type=str, help="Directory to output images in")
    args = parser.parse_args()
    # fmt: on

    out_size = tuple(int(s) for s in args.out_size.split(","))

    G_cls = get_generator_class(args.architecture)
    G = G_cls(
        model_file=args.model_file, output_size=out_size, strategy=args.resize_strategy, layer=args.resize_layer
    ).to(device)

    output_file = (
        f"{args.out_dir}/{Path(args.model_file).stem}_interpolation_{str(uuid4())[:8]}_smooth{args.smooth}.mp4"
    )

    with VideoWriter(output_file, out_size, args.fps) as video:
        for frame in random_interpolation(
            G=G, n_frames=args.n_frames, smooth=args.smooth, truncation=args.truncation, batch_size=args.batch_size
        ):
            video.write(frame.add(1).div(2).unsqueeze(0))
