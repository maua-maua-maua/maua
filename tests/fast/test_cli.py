"""The CLI must construct its full parser tree and parse real argvs without executing anything."""

import pytest

GROUPS = ["autoregressive", "diffusion", "super", "style"]


def get_parser():
    from maua.cli.entrypoint import argument_parser

    return argument_parser()


def test_parser_constructs():
    get_parser()


@pytest.mark.parametrize("group", GROUPS)
def test_group_help_exits_zero(group):
    with pytest.raises(SystemExit) as excinfo:
        get_parser().parse_args([group, "--help"])
    assert excinfo.value.code == 0


@pytest.mark.parametrize(
    "argv",
    [
        ["diffusion", "image", "--text", "a test prompt", "--sizes", "64,64"],
        ["diffusion", "video", "--init", "video.mp4", "--text", "a test prompt"],
        ["style", "image", "--content", "c.jpg", "--styles", "s.jpg"],
    ],
    ids=lambda argv: " ".join(argv[:2]),
)
def test_full_argv_parses(argv):
    args = get_parser().parse_args(argv)
    assert callable(getattr(args, "func", None)), "leaf command should bind a lazy dispatch function"
