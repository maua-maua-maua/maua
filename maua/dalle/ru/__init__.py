import argparse

from . import finetune, generate


def argument_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser(
        "generate", parents=[generate.argument_parser()], help="Generate images with RuDALL-E", add_help=False
    ).set_defaults(func=generate.main)
    subparsers.add_parser(
        "finetune",
        parents=[finetune.argument_parser()],
        help="Finetune RuDALL-E on a set of images (and captions)",
        add_help=False,
    ).set_defaults(func=finetune.main)
    return parser
