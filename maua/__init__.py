import argparse

from . import dalle, diffusion


def argument_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser(
        "dalle", parents=[dalle.argument_parser()], help="Generate images using DALL-E", add_help=False
    )
    subparsers.add_parser(
        "diffusion",
        parents=[diffusion.argument_parser()],
        help="Generate images using score-matching/diffusion models",
        add_help=False,
    )
    return parser
