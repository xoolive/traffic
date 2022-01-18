import pandas as pd

from ... import aixm_navaids


def points() -> pd.DataFrame:
    fra_role = [  # noqa: F841
        "FRA_ENTRY",
        "FRA_EXIT",
        "FRA_INTERMEDIATE",
        "FRA_DEPARTURE",
        "FRA_ARRIVAL",
    ]
    return (
        aixm_navaids.extensions.query("role in @fra_role")
        .merge(aixm_navaids.data, on="id")
        .drop(columns=["description"])
    )
