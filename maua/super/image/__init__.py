import argparse

from . import comparison, single  # , bulk
from .single import MODEL_NAMES, upscale


def argument_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser(
        "upscale",
        parents=[single.argument_parser()],
        help="Upscale images using a specific model",
        add_help=False,
    ).set_defaults(func=single.main)
    subparsers.add_parser(
        "comparison",
        parents=[comparison.argument_parser()],
        help="Run all of the models to compare their outputs",
        add_help=False,
    ).set_defaults(func=comparison.main)
    # subparsers.add_parser( # TODO
    #     "bulk",
    #     parents=[bulk.argument_parser()],
    #     help="Multi-GPU efficient inference for large batches of images",
    #     add_help=False,
    # ).set_defaults(func=bulk.main)
    return parser
