import argparse
from . import autoregressive, diffusion, style, super


def argument_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser(
        "autoregressive",
        parents=[autoregressive.argument_parser()],
        help="Generate images using autoregressive sampling (e.g. DALL-E)",
        add_help=False,
    )
    subparsers.add_parser(
        "diffusion",
        parents=[diffusion.argument_parser()],
        help="Generate images using score-matching / denoising diffusion models",
        add_help=False,
    )
    subparsers.add_parser(
        "super",
        parents=[super.argument_parser()],
        help="Upscale images using super resolution models",
        add_help=False,
    )
    subparsers.add_parser(
        "style",
        parents=[style.argument_parser()],
        help="Neural style transfer",
        add_help=False,
    )
    return parser
