import sys

from .data import airports as data_airports
from .data import navaids as data_navaids


def get_airports(*args):
    for airport in data_airports.search(args[0]):
        print(airport)

def get_navaids(*args):
    for navaid in data_navaids.search(args[0]):
        print(navaid)

cmd = {'airport': get_airports,
       'navaid': get_navaids,
       }


def main():
    command = sys.argv[1]
    cmd[command](*sys.argv[2:])
