import tempfile
from typing import Callable

import pytest

from traffic.core import Traffic
from traffic.data.samples import zurich_airport


@pytest.mark.parametrize(
    "data_ext,save_func",
    [
        (".csv", Traffic.to_csv),
        (".pkl", Traffic.to_pickle),
        (".parquet", Traffic.to_parquet),
        (".json", Traffic.to_json),
        # pr295 replaces pyarrow with fastparquet
        # (".feather", Traffic.to_feather),
    ],
)
def test_save_and_load(
    data_ext: str, save_func: Callable[[Traffic, str], None]
) -> None:
    for ext in ["", ".gzip", ".bz2", ".zip", ".xz"]:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = f"{tmpdir}/zurich{data_ext}{ext}"
            save_func(zurich_airport, file_path)
            traffic = Traffic.from_file(file_path)
            assert traffic is not None
