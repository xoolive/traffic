# %%
from pathlib import Path
import pandas as pd
import csv
from datetime import timedelta


def sectors_openings(
    path_file: Path = Path("../../sectors_LFBB/2022-07-BORD/2022-07-14_BORD"),
) -> pd.DataFrame:
    dict = {"start": [], "stop": [], "group": [], "decomposition": []}
    with open(path_file) as f:
        reader = csv.reader(f, delimiter=" ")
        for i, lines in enumerate(reader):
            lines = list(filter(None, lines))
            if lines[0] == "02":
                date_str = lines[1]
                ts_base = pd.Timestamp(f"{date_str} 00:00", tz="utc")
            if lines[0] == "10":
                start = lines[2]
                stop = lines[3]
            elif lines[0] == "11":
                if start != stop:
                    ts_start = pd.Timestamp(
                        ts_base + timedelta(minutes=int(start))
                    ).tz_convert("utc")
                    ts_stop = pd.Timestamp(
                        ts_base + timedelta(minutes=int(stop))
                    ).tz_convert("utc")
                    group = lines[2]
                    decomp = lines[5::]
                    dict["start"].append(ts_start)
                    dict["stop"].append(ts_stop)
                    dict["start"]
                    dict["group"].append(group)
                    dict["decomposition"].append(decomp)
        result = pd.DataFrame(dict)
        result.replace("?????", None)
    return result
    # %%


dic = sectors_openings()

# %%
