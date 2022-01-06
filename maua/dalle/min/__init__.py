import argparse

from . import generate


def argument_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser(
        "generate", parents=[generate.argument_parser()], help="Generate images with MinDALL-E", add_help=False
    ).set_defaults(func=generate.main)
    return parser
