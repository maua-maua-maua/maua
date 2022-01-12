import argparse

from . import image, video


def argument_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser(
        "image",
        parents=[image.argument_parser()],
        help="Upscale images",
        add_help=False,
    )
    subparsers.add_parser(
        "video",
        parents=[video.argument_parser()],
        help="Upscale videos (either in resolution or frame rate)",
        add_help=False,
    )
    return parser
