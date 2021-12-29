import argparse
from pathlib import Path
from typing import Tuple

import torch

from patches.base import get_patch_from_file
from render import get_output_class
from util import write_video

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_audiovisal_from_patch(
    audio_file: str,
    model_file: str,
    patch_file: str,
    renderer: str,
    fps: float,
    out_size: Tuple[int],
    resize_strategy: str,
    resize_layer: int,
) -> torch.Tensor:
    patch_class = get_patch_from_file(patch_file)
    patch = patch_class(
        model_file,
        audio_file,
        fps=fps,
        offset=0,
        duration=-1,
        output_size=out_size,
        resize_strategy=resize_strategy,
        resize_layer=resize_layer,
    )

    patch.process_audio()
    mapper_inputs = patch.process_mapper_inputs()
    mapped_inputs = patch.mapper(**mapper_inputs)
    synthesizer_inputs = patch.process_synthesizer_inputs(mapped_inputs)
    video = get_output_class(renderer)(patch.synthesizer, synthesizer_inputs)
    final = patch.process_outputs(video)

    return final


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_file", required=True, type=str, help="Path to audio file")
    parser.add_argument("--model_file", required=True, type=str, help="Path to .pkl file containing the model to use")
    parser.add_argument("--patch_file", default="patches/examples/default.py", type=str, help="The file which defines the audio-reactive modulations of the GANs inputs")
    parser.add_argument("--renderer", default="pytorch", type=str, help="The method used to render your video")
    parser.add_argument("--fps", default=24, type=float, help="Frames per second of output video")
    parser.add_argument("--out_size", default="1024,1024", type=str, help="Desired width,height of output image: e.g. 1920,1080 or 720,1280")
    parser.add_argument("--resize_strategy", default="pad-zero", choices=["pad-zero", "stretch"], type=str, help="Strategy used to resize (in feature space) to achieve desired output resolution")
    parser.add_argument("--resize_layer", default=0, choices=list(range(15)), type=int, help="Which layer in the network to perform resizing at. Higher values are closer to resizing output pixels directly. Lower values have larger rounding increments (i.e. less flexible possible output sizes)")
    parser.add_argument("--out_dir", default="./workspace/", type=str, help="Directory to output video in")
    args = parser.parse_args()
    args.out_size = tuple(int(s) for s in args.out_size.split(","))
    # fmt: on

    video = generate_audiovisal_from_patch(
        audio_file=args.audio_file,
        model_file=args.model_file,
        patch_file=args.patch_file,
        renderer=args.renderer,
        fps=args.fps,
        out_size=args.out_size,
        resize_strategy=args.resize_strategy,
        resize_layer=args.resize_layer,
    )

    checkpoint_name = Path(args.model_file.replace("/network-snapshot", "")).stem
    output_file = f"{args.out_dir}/{checkpoint_name}_{args.resize_strategy}.mp4"
    write_video(video, output_file, args.fps)
