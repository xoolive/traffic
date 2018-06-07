import sys

from .data import airports as data_airports
from .data import navaids as data_navaids
from .data import aircraft as data_aircraft


def get_airports(*args):
    for airport in data_airports.search(args[0]):
        print(airport)


def get_navaids(*args):
    for navaid in data_navaids.search(args[0]):
        print(navaid)


def get_aircraft(*args):
    if args[0] == "get":
        print(data_aircraft[args[1]])
    elif args[0] == "operator":
        print(data_aircraft.operator(" ".join(args[1:])))
    elif args[0] == "stats":
        print(data_aircraft.stats(" ".join(args[1:])))
    else:
        raise RuntimeError("Usage: traffic aircraft [get|operator|stats]")


cmd = {"airport": get_airports, "navaid": get_navaids, "aircraft": get_aircraft}


def main():
    command = sys.argv[1]
    cmd[command](*sys.argv[2:])
