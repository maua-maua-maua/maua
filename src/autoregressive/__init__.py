import argparse

from . import min_dalle, rqvaeformer, ru_dalle


def argument_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser(
        "ru", parents=[ru_dalle.argument_parser()], help="Generate images with RuDALL-E", add_help=False
    )
    subparsers.add_parser(
        "min", parents=[min_dalle.argument_parser()], help="Generate images with MinDALL-E", add_help=False
    )
    subparsers.add_parser(
        "rqvae", parents=[rqvaeformer.argument_parser()], help="Generate images with RQVAE Transformer", add_help=False
    )
    return parser
