import sys

from .data import airports as data_airports


def get_airports(*args):
    for airport in data_airports.search(args[0]):
        print(airport)

cmd = {'airport': get_airports,
       }


def main():
    command = sys.argv[1]
    cmd[command](*sys.argv[2:])
