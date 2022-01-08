import argparse

from . import image, video


def argument_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser(
        "image",
        parents=[image.argument_parser()],
        help="Generate images with neural style transfer",
        add_help=False,
    ).set_defaults(func=image.main)
    subparsers.add_parser(
        "video", parents=[video.argument_parser()], help="Generate videos with neural style transfer", add_help=False
    ).set_defaults(func=video.main)
    return parser
