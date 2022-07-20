import argparse

from . import main_function


def argument_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser(
        "image",
        parents=[image()],
        help="Upscale images",
        add_help=False,
    )
    subparsers.add_parser(
        "video",
        parents=[video()],
        help="Upscale videos (either in resolution or frame rate)",
        add_help=False,
    )
    return parser


def image():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser(
        "upscale",
        parents=[single()],
        help="Upscale images using a specific model",
        add_help=False,
    ).set_defaults(func=main_function("maua.super.image.single"))
    subparsers.add_parser(
        "comparison",
        parents=[comparison()],
        help="Run all of the models to compare their outputs",
        add_help=False,
    ).set_defaults(func=main_function("maua.super.image.comparison"))
    # subparsers.add_parser( # TODO
    #     "bulk",
    #     parents=[bulk()],
    #     help="Multi-GPU efficient inference for large batches of images",
    #     add_help=False,
    # ).set_defaults(func=main_function("maua.super.image.bulk"))
    return parser


def single():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument(
        "--model_name", default="latent-diffusion", help="see --model-help for options"
    )  # TODO --model-help
    parser.add_argument("--postdownsample", default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="output/")
    return parser


def comparison():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="output/")
    return parser


def video():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser(
        "frame-by-frame",
        parents=[frame_by_frame()],
        help="Increase video resolution by upscaling each frame individually",
        add_help=False,
    ).set_defaults(func=main_function("maua.super.video.frame_by_frame"))
    subparsers.add_parser(
        "framerate",
        parents=[framerate()],
        help="Increase video frame rate",
        add_help=False,
    ).set_defaults(func=main_function("maua.super.video.framerate"))
    # subparsers.add_parser( # TODO
    #     "spatiotemporal",
    #     parents=[spatiotemporal()],
    #     help="Increase video resolution",
    #     add_help=False,
    # ).set_defaults(func=main_function("maua.super.video.spatiotemporal"))
    return parser


def frame_by_frame():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_files", nargs="+")
    parser.add_argument(
        "--model_name", default="latent-diffusion", help="see --model-help for options"
    )  # TODO --model-help
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="output/")
    return parser


def framerate():
    # fmt: off
    parser = argparse.ArgumentParser(description="The output frame rate can be calculated by: original_fps * interpolation_factor / slower / decimate")
    parser.add_argument("video_files", nargs="+")
    parser.add_argument("--model_name", default="RIFE-2.3", help='see --model-help for options')  # TODO --model-help
    parser.add_argument("-if", "--interpolation_factor", type=int, default=2, help="Factor to increase framerate by")
    parser.add_argument("-s", "--slower", type=int, default=1, help="Factor to decrease output framerate by")
    parser.add_argument("-d", "--decimate", type=int, default=2, help="Alternative to slower that samples every -d'th frame. ")
    parser.add_argument("--no-fp16", action="store_true", help="FP16 reduces memory usage and increases speed on tensor cores (disable for CPU)")
    parser.add_argument("--device", default='cuda')
    parser.add_argument("--out_dir", default="output/")
    # fmt: on
    return parser
