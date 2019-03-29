import argparse
import json
from pathlib import Path
from typing import List, Optional

# Convenient one-liner for getting sector types
# set(s.type for s in sectors.parse(".*"))


def export_json(patterns: List[str], output_file: Optional[Path]):
    from traffic.data import nm_airspaces

    if output_file is None:
        for pattern in patterns:
            for sector in nm_airspaces.parse(pattern):
                print(f"{sector.name}/{sector.type}")
        return

    with open(output_file.as_posix(), "w") as fh:
        json.dump(
            [
                s.export_json()
                for pattern in patterns
                for s in nm_airspaces.search(pattern)
            ],
            fh,
            sort_keys=False,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export sectors to json")

    parser.add_argument(
        "-o", dest="output_file", type=Path, help="name for output json file"
    )

    parser.add_argument("patterns", nargs="+", help="patterns to match")

    args = parser.parse_args()
    export_json(**vars(args))
