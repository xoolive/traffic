import argparse
import logging
import sys

from . import config, data, decode, gui, makeapp, opensky, show

cmd = {
    "config": config.main,
    "data": data.main,
    "decode": decode.main,
    "config": config.main,
    "gui": gui.main,
    "makeapp": makeapp.main,
    "opensky": opensky.main,
    "show": show.main
}


def main():

    parser = argparse.ArgumentParser(
        description="traffic command-line interface",
        epilog="For specific help about each command, type traffic command -h",
    )

    parser.add_argument(
        "command", help=f"choose among: {', '.join(cmd.keys())}"
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="all arguments to dispatch to command",
    )

    args = parser.parse_args()
    fun = cmd.get(args.command, None)

    if fun is None:
        return parser.print_help()

    return fun(args.args)
