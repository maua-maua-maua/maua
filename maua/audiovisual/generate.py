import argparse
from pathlib import Path
from typing import Tuple
from uuid import uuid4

import torch

from maua.ops.video import write_video

from .patches.base import get_patch_from_file
from .render import get_output_class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def generate_audiovisal_from_patch(
    audio_file: str,
    model_file: str,
    patch_file: str,
    patch_name: str,
    renderer: str,
    renderer_kwargs: dict,
    fps: float,
    out_size: Tuple[int],
    resize_strategy: str,
    resize_layer: int,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, int]]:

    patch = get_patch_from_file(patch_file, patch_name)(
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

    postprocess = lambda video: patch.force_output_size(patch.process_outputs(video))

    renderer_kwargs["fps"] = patch.fps
    renderer_kwargs["audio_file"] = patch.audio_file
    video = get_output_class(renderer)(**renderer_kwargs)(patch.synthesizer, synthesizer_inputs, postprocess)

    return video, (patch.audio, patch.sr)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_file", required=True, type=str, help="Path to audio file")
    parser.add_argument("--model_file", required=True, type=str, help="Path to .pkl file containing the model to use")
    parser.add_argument("--patch_file", default="patches/examples/default.py", type=str, help="The file which defines the audio-reactive modulations of the GANs inputs")
    parser.add_argument("--patch_name", default=None, type=str, help="Which patch class to use (if there are multiple in the file)")
    parser.add_argument("--renderer", default="ffmpeg", type=str, help="The method used to render your video")
    parser.add_argument("--ffmpeg_preset", default="fast", type=str, help="If rendering with FFMPEG, the preset for video encoding. Slower is higher quality and smaller file size. For options see: https://trac.ffmpeg.org/wiki/Encode/H.264")
    parser.add_argument("--fps", default=24, type=float, help="Frames per second of output video")
    parser.add_argument("--out_size", default="1024,1024", type=str, help="Desired width,height of output image: e.g. 1920,1080 or 720,1280")
    parser.add_argument("--resize_strategy", default="pad-zero", type=str, help="Strategy used to resize (in feature space) to achieve desired output resolution")
    parser.add_argument("--resize_layer", default=0, choices=list(range(18)), type=int, help="Which layer in the network to perform resizing at. Higher values are closer to resizing output pixels directly. Lower values have larger rounding increments (i.e. less flexible possible output sizes)")
    parser.add_argument("--out_dir", default="./output/", type=str, help="Directory to output video in")
    parser.add_argument("--unique", action='store_true', help="Whether to add a unique identifier to the filename")
    args = parser.parse_args()
    # fmt: on

    checkpoint_name = Path(args.model_file.replace("/network-snapshot", "")).stem
    output_file = f"{args.out_dir}/{Path(args.audio_file).stem}_{checkpoint_name}_{args.resize_strategy}_{args.out_size.replace(',', 'x')}.mp4"
    if args.unique:
        output_file = output_file.replace(".mp4", f"-{str(uuid4())[:6]}.mp4")
    args.out_size = tuple(int(s) for s in args.out_size.split(","))

    if args.renderer == "ffmpeg":
        renderer_kwargs = dict(output_file=output_file, ffmpeg_preset=args.ffmpeg_preset)

    video, (audio, sr) = generate_audiovisal_from_patch(
        audio_file=args.audio_file,
        model_file=args.model_file,
        patch_file=args.patch_file,
        patch_name=args.patch_name,
        renderer=args.renderer,
        renderer_kwargs=renderer_kwargs,
        fps=args.fps,
        out_size=args.out_size,
        resize_strategy=args.resize_strategy,
        resize_layer=args.resize_layer,
    )

    if args.renderer == "memmap":
        write_video(tensor=video, output_file=output_file, fps=args.fps, audio_file=args.audio_file)
