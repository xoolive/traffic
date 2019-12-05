import argparse
import importlib
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import pkg_resources


def dispatch_open(filename: Path):
    if sys.platform.startswith("darwin"):
        subprocess.call(("open", filename))
    elif os.name == "nt":  # For Windows
        os.startfile(filename)  # type: ignore
    elif os.name == "posix":  # For Linux, Mac, etc.
        subprocess.call(("xdg-open", filename))


def import_submodules(package, recursive=True):
    """ Import all submodules of a module, recursively, including subpackages

    :param package: package (name or actual module)
    :type package: str | module
    :rtype: dict[str, types.ModuleType]
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for _loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + "." + name
        results[name] = importlib.import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    for entry_point in pkg_resources.iter_entry_points("traffic.console"):
        handle = entry_point.load()
        results[entry_point.name] = handle
    return results


def main():

    cmd = import_submodules(__name__, recursive=False)

    parser = argparse.ArgumentParser(
        description="traffic command-line interface",
        epilog="For specific help about each command, type traffic command -h",
    )

    parser.add_argument("command", help=f"among: {', '.join(cmd.keys())}")
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="all arguments to dispatch to command",
    )

    args = parser.parse_args()

    mod = cmd.get(args.command, None)

    if mod is None:
        return parser.print_help()

    return mod.main(args.args)
