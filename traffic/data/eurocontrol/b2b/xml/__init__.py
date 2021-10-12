from pathlib import Path
from typing import DefaultDict


class RequestsDict(DefaultDict[str, str]):
    def __missing__(self, key: str) -> str:
        candidate_file = Path(__file__).parent / (key + ".xml")
        if not candidate_file.exists():
            raise KeyError(key)
        with candidate_file.open("r") as fh:
            result = "\n".join(fh.readlines())
        self[key] = result
        return result


REQUESTS = RequestsDict()
