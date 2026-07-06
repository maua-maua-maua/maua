"""The CLI must construct its full parser tree and parse real argvs without executing anything."""

import importlib
import re
from pathlib import Path

import pytest

GROUPS = ["autoregressive", "diffusion", "super", "style"]

CLI_DIR = Path(__file__).resolve().parents[2] / "maua" / "cli"


def leaf_module_targets():
    """Every active `main_function("module.path")` wired into the CLI (commented-out lines excluded)."""
    targets = set()
    for cli_file in CLI_DIR.glob("*.py"):
        for line in cli_file.read_text().splitlines():
            if line.lstrip().startswith("#"):
                continue
            m = re.search(r'main_function\("([^"]+)"\)', line)
            if m:
                targets.add(m.group(1))
    return sorted(targets)


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


@pytest.mark.parametrize("module_path", leaf_module_targets())
def test_leaf_dispatch_target_has_main(module_path):
    """Each `main_function(...)` target must import and expose a callable `main` — the lazy
    dispatch calls `import_module(name).main(args)`, so a typo or missing main only surfaces here."""
    module = importlib.import_module(module_path)
    assert callable(getattr(module, "main", None)), f"{module_path} has no callable main()"
