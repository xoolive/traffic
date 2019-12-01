from collections import defaultdict
from pathlib import Path


class RequestsDict(defaultdict):
    def __missing__(self, key):
        candidate_file = Path(__file__).parent / (key + ".xml")
        if not candidate_file.exists():
            raise KeyError(key)
        with candidate_file.open("r") as fh:
            result = "\n".join(fh.readlines())
        self[key] = result
        return result


REQUESTS = RequestsDict()
