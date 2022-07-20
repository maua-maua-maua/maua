#!python

from . import argument_parser


def main():
    args = argument_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
