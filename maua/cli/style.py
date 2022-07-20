import argparse

# from ..optimizers import OPTIMIZERS
from . import main_function


def argument_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.add_parser(
        "image",
        parents=[image()],
        help="Generate images with neural style transfer",
        add_help=False,
    ).set_defaults(func=main_function("maua.style.image"))
    subparsers.add_parser(
        "video", parents=[video()], help="Generate videos with neural style transfer", add_help=False
    ).set_defaults(func=main_function("maua.style.video"))
    return parser


def image():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)  # description=transfer.__doc__
    parser.add_argument("--content")
    parser.add_argument("--styles", nargs="+")
    parser.add_argument("--init_img", type=str, default=None)
    parser.add_argument("--init_type", default="content", choices=['content', 'random', 'init_img'])
    parser.add_argument("--match_hist", default="avg", choices=['avg', 'False'])
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--parameterization", default="rgb", choices=["rgb", "vqgan"])
    parser.add_argument("--perceptor", default="kbc-vgg19", choices=["kbc-vgg19" ,"pgg-vgg19", "pgg-vgg16", "pgg-prune", "pgg-nyud", "pgg-fcn32s", "pgg-sod", "pgg-nin"])
    parser.add_argument("--perceptor_kwargs", nargs="*", default=[])
    parser.add_argument("--optimizer", default="LBFGS", help="see --optimizer-help") # TODO --optimizer-help
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--optimizer_kwargs", nargs="*", default=[])
    parser.add_argument("--n_iters", type=int, default=512)
    parser.add_argument("--content_weight", type=float, default=1)
    parser.add_argument("--style_weight", type=float, default=50)
    parser.add_argument("--tv_weight", type=float, default=100)
    parser.add_argument("--style_scale", type=float, default=1)
    parser.add_argument("--device", default='cuda')
    # fmt: on
    return parser


def video():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)  # description=transfer.__doc__
    parser.add_argument("--content")
    parser.add_argument("--styles", nargs="+")
    parser.add_argument("--init_type", default="content", choices=["content", "random", "prev_warped"])
    parser.add_argument("--match_hist", default="avg", choices=["avg", False])
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--parameterization", default="rgb", choices=["rgb", "vqgan"])
    parser.add_argument("--perceptor", default="kbc-vgg19", choices=["kbc-vgg19", "pgg-vgg19", "pgg-vgg16", "pgg-prune", "pgg-nyud", "pgg-fcn32s", "pgg-sod", "pgg-nin"])
    parser.add_argument("--perceptor_kwargs", default={})
    parser.add_argument("--optimizer", default="LBFGS", help="see --optimizer-help") # TODO --optimizer-help
    parser.add_argument("--lr", type=float,default=0.5)
    parser.add_argument("--optimizer_kwargs", default={})
    parser.add_argument("--flow_models", nargs="+", default=["farneback"], choices=["farneback", "spynet", "pwc", "liteflownet", "unflow"])
    parser.add_argument("--n_iters", type=int, default=512)
    parser.add_argument("--n_passes", type=int, default=16)
    parser.add_argument("--temporal_loss_after", type=int, default=2)
    parser.add_argument("--blend_factor", type=float, default=1)
    parser.add_argument("--content_weight", type=float, default=1)
    parser.add_argument("--style_weight", type=float, default=5000)
    parser.add_argument("--tv_weight", type=float, default=10)
    parser.add_argument("--temporal_weight", type=float, default=100)
    parser.add_argument("--style_scale", type=float, default=1)
    parser.add_argument("--start_random_frame", action="store_true")
    parser.add_argument("--device", default='cuda')
    parser.add_argument("--save_intermediate", action="store_true")
    parser.add_argument("--fps", type=float, default=24)
    parser.add_argument("--out_dir", default="output/")
    # fmt: on
    return parser
