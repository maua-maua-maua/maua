import argparse

from . import frame_by_frame, framerate  # , spatiotemporal


def argument_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser(
        "frame-by-frame",
        parents=[frame_by_frame.argument_parser()],
        help="Increase video resolution by upscaling each frame individually",
        add_help=False,
    ).set_defaults(func=frame_by_frame.main)
    subparsers.add_parser(
        "framerate",
        parents=[framerate.argument_parser()],
        help="Increase video frame rate",
        add_help=False,
    ).set_defaults(func=framerate.main)
    # subparsers.add_parser( # TODO
    #     "spatiotemporal",
    #     parents=[spatiotemporal.argument_parser()],
    #     help="Increase video resolution",
    #     add_help=False,
    # ).set_defaults(func=spatiotemporal.main)
    return parser
