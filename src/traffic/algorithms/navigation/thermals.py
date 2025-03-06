from typing import Iterator

from ...core import Flight


class GliderThermal:
    """Detects thermals for gliders."""

    def apply(self, flight: Flight) -> Iterator[Flight]:
        all_segments = (
            flight.unwrap()
            .diff("track_unwrapped")
            .agg_time(
                "1 min",
                vertical_rate="max",
                track_unwrapped_diff="median",
            )
            .abs(track_unwrapped_diff_median="track_unwrapped_diff_median")
            .query("vertical_rate_max > 2 and track_unwrapped_diff_median > 5")
        )
        if all_segments is not None:
            yield from all_segments.split("1 min")
