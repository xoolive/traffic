import argparse
import importlib
import os
import pkgutil
import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Dict


def dispatch_open(filename: Path) -> None:
    if sys.platform.startswith("darwin"):
        subprocess.call(("open", filename))
    elif os.name == "nt":  # For Windows
        os.startfile(filename)  # type: ignore
    elif os.name == "posix":  # For Linux, Mac, etc.
        subprocess.call(("xdg-open", filename))


def import_submodules(package: Any, recursive: bool = True) -> Dict[str, Any]:
    """Import all submodules of a module, recursively, including subpackages

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
    try:
        # https://docs.python.org/3/library/importlib.metadata.html#entry-points
        ep = entry_points(group="traffic.console")
    except TypeError:
        ep = {
            m.name: m  # type: ignore
            for m in entry_points().get("traffic.console", [])
        }
    for entry_point in ep:
        handle = entry_point.load()
        results[entry_point.name] = handle
    return results


def main() -> Any:
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

    mod.main(args.args)
