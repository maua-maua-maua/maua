import argparse

from maua.cli import main_function
from maua.diffusion.finetune_stable import argument_parser as finetune_stable_argument_parser
from maua.diffusion.image import argument_parser as image_argument_parser
from maua.diffusion.klmc2_animation import argument_parser as klmc2_animation_argument_parser
from maua.diffusion.temporalvideo_hf import argument_parser as temporalvideo_hf_argument_parser
from maua.diffusion.video import argument_parser as video_argument_parser


def argument_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser(
        "image",
        parents=[image_argument_parser()],
        help="Generate images with diffusion models",
        add_help=False,
    ).set_defaults(func=main_function("maua.diffusion.image"))
    subparsers.add_parser(
        "video",
        parents=[video_argument_parser()],
        help="Stylize videos with diffusion and optical flow",
        add_help=False,
    ).set_defaults(func=main_function("maua.diffusion.video"))
    subparsers.add_parser(
        "finetune-stable",
        parents=[finetune_stable_argument_parser()],
        help="Fine-tune Stable Diffusion on a directory of images",
        add_help=False,
    ).set_defaults(func=main_function("maua.diffusion.finetune_stable"))
    subparsers.add_parser(
        "video-hf",
        parents=[temporalvideo_hf_argument_parser()],
        help="HF ControlNet / img2img temporal video stylization",
        add_help=False,
    ).set_defaults(func=main_function("maua.diffusion.temporalvideo_hf"))
    subparsers.add_parser(
        "klmc2",
        parents=[klmc2_animation_argument_parser()],
        help="KLMC2 animation sampling (k-diffusion)",
        add_help=False,
    ).set_defaults(func=main_function("maua.diffusion.klmc2_animation"))
    return parser
