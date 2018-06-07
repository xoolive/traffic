import json
from collections import UserDict
from pathlib import Path


class Cache(UserDict):

    def __init__(self, cachedir: Path) -> None:
        self.cachedir = cachedir
        if not self.cachedir.exists():
            self.cachedir.mkdir(parents=True)
        super().__init__()

    def __missing__(self, hashcode: str):
        filename = self.cachedir / f"{hashcode}.json"
        if filename.exists():
            response = json.loads(filename.read_text())
            # Do not overwrite the file!
            super().__setitem__(hashcode, response)
            return response

    def __setitem__(self, hashcode: str, data):
        super().__setitem__(hashcode, data)
        filename = self.cachedir / f"{hashcode}.json"
        with filename.open('w') as fh:
            fh.write(json.dumps(data))
