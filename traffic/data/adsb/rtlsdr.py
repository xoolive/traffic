from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, TextIO, Tuple

from pyModeS.extra.rtlreader import RtlReader

if TYPE_CHECKING:
    from .decode import Decoder


class MyRtlReader(RtlReader):
    def __init__(
        self,
        decoder: "Decoder",
        fh: Optional[TextIO] = None,
        uncertainty: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.count = 0
        self.fh = fh
        self.decoder = decoder
        self.uncertainty = uncertainty

    def encode_message(
        self, msg: str, time: float
    ) -> Optional[Tuple[float, str]]:

        now = datetime.now(timezone.utc)
        secs_total = (
            now - now.replace(hour=0, minute=0, second=0, microsecond=0)
        ).total_seconds()

        secs = int(secs_total)
        nanosecs = int((secs_total - secs) * 1e9)
        t = (secs << 30) + nanosecs

        if len(msg) == 28:
            return time, f"1a33{t:012x}00{msg.lower()}"
        if len(msg) == 14:
            return time, f"1a32{t:012x}00{msg.lower()}"
        if len(msg) == 2:
            return time, f"1a31{t:012x}00{msg.lower()}"
        return None

    def handle_messages(self, messages):

        for msg, t in messages:

            self.count += 1

            if self.fh is not None:
                encoded = self.encode_message(msg, t)
                if encoded is not None:
                    self.fh.write("{},{}\n".format(*encoded))

            if self.count & 127 == 127:
                self.decoder.redefine_reference(t)

            self.decoder.process(
                datetime.fromtimestamp(t, timezone.utc),
                msg,
                uncertainty=self.uncertainty,
            )
