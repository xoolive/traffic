import warnings
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, NoReturn, Optional, Set, Type, TypeVar
from xml.dom import minidom
from xml.etree import ElementTree

import pandas as pd

from ....core.mixins import DataFrameMixin
from ....core.time import timelike, to_datetime
from .reply import B2BReply
from .xml import REQUESTS

default_regulation_fields: Set[str] = {
    # "applicability",
    # "autolink",
    # "measureCherryPicked",
    # "initialConstraints",
    # "linkedRegulations",
    "location",
    "protectedLocation",
    # "reason",
    # "remark",
    "regulationState",
    # "supplementaryConstraints",
    # "lastUpdate",
    # "noDelayWindow",
    # "updateCapacityRequired",
    # "updateTVActivationRequired",
    # "externallyEditable",
    "subType",
    # "delayTVSet",
    # "createdByFMP",
    # "sourceHotspot",
    # "mcdmRequired",
    # "dataId",
    # "scenarioReference",
    # "delayConfirmationThreshold",
}

# https://github.com/python/mypy/issues/2511
RegulationListTypeVar = TypeVar("RegulationListTypeVar", bound="RegulationList")


class RegulationList(DataFrameMixin, B2BReply):
    def __init__(self, *args, **kwargs):
        if len(args) == 0 and "data" not in kwargs:
            super().__init__(data=None, **kwargs)
        else:
            super().__init__(*args, **kwargs)

    @classmethod
    def fromB2BReply(cls, r: B2BReply):
        assert r.reply is not None
        return cls.fromET(r.reply)

    @classmethod
    def fromET(
        cls: Type[RegulationListTypeVar], tree: ElementTree.Element
    ) -> RegulationListTypeVar:
        instance = cls()
        instance.reply = tree
        instance.build_df()
        return instance

    def build_df(self) -> None:
        assert self.reply is not None

        self.data = pd.DataFrame.from_records(
            [
                {**{p.tag: p.text for p in elt if p.text is not None}}
                for elt in self.reply.findall("data/regulations/item")
            ]
        )


class Measures:
    def list_regulation(
        self,
        start: timelike,
        stop: Optional[timelike] = None,
        tvs: List[str] = [],
        fields: List[str] = [],
    ) -> Optional[RegulationList]:

        start = to_datetime(start)
        if stop is not None:
            stop = to_datetime(stop)
        else:
            stop = start + timedelta(days=1)

        data = REQUESTS["RegulationListRequest"].format(
            send_time=datetime.now(timezone.utc),
            start=start,
            stop=stop,
            requestedRegulationFields=(
                "<requestedRegulationFields>\n"
                + "\n".join(
                    f"<item>{field}</item>"
                    for field in default_regulation_fields.union(fields)
                )
                + "</requestedRegulationFields>"
            ),
            tvs=(
                "<tvs>\n"
                + "\n".join(f"<item>{tv}</item>" for tv in tvs)
                + "</tvs>"
            ),
        )
        print(data)
        rep = self.post(data)  # type: ignore
        return RegulationList.fromB2BReply(rep)
