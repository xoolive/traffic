import sys
import argparse
from pathlib import Path


def get_airports(*args):
    from .data import airports as data_airports
    for airport in data_airports.search(args[0]):
        print(airport)


def get_navaids(*args):
    from .data import navaids as data_navaids
    for navaid in data_navaids.search(args[0]):
        print(navaid)


def get_aircraft(*args):
    from .data import aircraft as data_aircraft
    if args[0] == "get":
        print(data_aircraft[args[1]])
    elif args[0] == "operator":
        print(data_aircraft.operator(" ".join(args[1:])))
    elif args[0] == "stats":
        print(data_aircraft.stats(" ".join(args[1:])))
    else:
        raise RuntimeError("Usage: traffic aircraft [get|operator|stats]")


def decode(*args):
    from .data.adsb.decode import Decoder
    parser = argparse.ArgumentParser()

    parser.add_argument("file", help="path to the file to decode", type=Path)
    parser.add_argument(
        "reference", help="reference airport for decoding surface position"
    )
    parser.add_argument("-o", "--output", help="output pickle file", type=Path)

    args = parser.parse_args(args)
    decoder = Decoder.from_file(args.file, args.reference)
    decoder.traffic.to_pickle(
        args.output
        if args.output is not None
        else args.file.with_suffix(".pkl")
    )

def launch_gui(*args):
    from traffic.qtgui import layout
    layout.main()

def config(*args):
    from . import edit_config
    edit_config()

cmd = {
    "airport": get_airports,
    "navaid": get_navaids,
    "aircraft": get_aircraft,
    "decode": decode,
    'config': config,
    'gui': launch_gui,
}


def main():
    command = sys.argv[1]
    cmd[command](*sys.argv[2:])
