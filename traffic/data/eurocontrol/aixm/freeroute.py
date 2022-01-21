import pandas as pd

from ... import aixm_navaids


def points() -> pd.DataFrame:
    return (
        aixm_navaids.extensions.query("role.str.startswith('FRA')")
        .merge(aixm_navaids.data, on="id")
        .drop(columns=["description"])
    )
