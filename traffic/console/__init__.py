import argparse
import logging
import sys

from . import data, decode, gui, makeapp, opensky
from .. import edit_config

cmd = {
    'data': data.main,
    "decode": decode.main,
    "config": edit_config,
    "gui": gui.main,
    "makeapp": makeapp.main,
    'opensky': opensky.main
}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="traffic command-line interface",
        epilog="For specific help about each command, type traffic command -h"
    )

    parser.add_argument("command", help=f"choose among: {', '.join(cmd.keys())}")
    parser.add_argument(
        "args", nargs=argparse.REMAINDER,
        help="all arguments to dispatch to command"
    )

    args = parser.parse_args()
    fun = cmd.get(args.command, None)

    if fun is None:
        return parser.print_help()

    return fun(args.args)
