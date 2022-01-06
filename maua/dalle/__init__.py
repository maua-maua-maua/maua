import argparse

from . import min, ru


def argument_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser("ru", parents=[ru.argument_parser()], help="Generate images with RuDALL-E", add_help=False)
    subparsers.add_parser("min", parents=[min.argument_parser()], help="Generate images with MinDALL-E", add_help=False)
    return parser
