# %%

from lxml import etree
from pathlib import Path
import os
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
import pandas as pd
from traffic.data import aixm_navaids

path = Path("../../../../../AIRAC_2207/")

# %%
# PARSER DE ROUTE.BASELINE
all_points: Dict[str, Dict[str, Any]] = {}
extensions: List[Dict[str, Any]] = []

dirname = path

ns: Dict[str, str] = dict()

for _, (key, value) in etree.iterparse(
    (dirname / "Route.BASELINE").as_posix(),
    events=["start-ns"],
):
    ns[key] = value

points = etree.parse((dirname / "Route.BASELINE").as_posix())

for point in points.findall("adrmsg:hasMember/aixm:Route", ns):
    identifier = point.find("gml:identifier", ns)
    assert identifier is not None
    assert identifier.text is not None

    designatorPrefix = point.find(
        "aixm:timeSlice/aixm:RouteTimeSlice/aixm:designatorPrefix",
        ns,
    )

    designatorSecondLetter = point.find(
        "aixm:timeSlice/aixm:RouteTimeSlice/aixm:designatorSecondLetter", ns
    )

    designatorNumber = point.find(
        "aixm:timeSlice/aixm:RouteTimeSlice/aixm:designatorNumber", ns
    )

    multipleIdentifier = point.find(
        "aixm:timeSlice/aixm:RouteTimeSlice/aixm:multipleIdentifier", ns
    )

    beginPosition = point.find(
        "aixm:timeSlice/aixm:RouteTimeSlice/gml:validTime/gml:TimePeriod/gml:beginPosition",
        ns,
    )

    endPosition = point.find(
        "aixm:timeSlice/aixm:RouteTimeSlice/gml:validTime/gml:TimePeriod/gml:endPosition",
        ns,
    )

    designatorPrefix_str = (
        designatorPrefix.text if designatorPrefix is not None else None
    )
    designatorSecondLetter_str = (
        designatorSecondLetter.text
        if designatorSecondLetter is not None
        else None
    )
    designatorNumber_str = (
        designatorNumber.text if designatorNumber is not None else None
    )
    multipleIdentifier_str = (
        multipleIdentifier.text if multipleIdentifier is not None else None
    )
    beginPosition_str = (
        beginPosition.text if beginPosition is not None else None
    )

    endPosition_str = endPosition.text if endPosition is not None else None

    all_points[identifier.text] = {
        "identifier": identifier.text,
        "prefix": designatorPrefix_str,
        "secondLetter": designatorSecondLetter_str,
        "number": designatorNumber_str,
        "multipleIdentifier": multipleIdentifier_str,
        "beginPosition": beginPosition_str,
        "endPosition": endPosition_str,
    }

data_routes = pd.DataFrame.from_records(point for point in all_points.values())

# %% PARSER DE SEGMENTS

all_points_seg: Dict[str, Dict[str, Any]] = {}
extensions_seg: List[Dict[str, Any]] = []

dirname = path

ns_seg: Dict[str, str] = dict()

for _, (key, value) in etree.iterparse(
    (dirname / "RouteSegment.BASELINE").as_posix(),
    events=["start-ns"],
):
    ns_seg[key] = value

    points = etree.parse((dirname / "RouteSegment.BASELINE").as_posix())

for point in points.findall("adrmsg:hasMember/aixm:RouteSegment", ns_seg):
    identifier = point.find("gml:identifier", ns_seg)
    assert identifier is not None
    assert identifier.text is not None

    beginPosition = point.find(
        "aixm:timeSlice/aixm:RouteSegmentTimeSlice/gml:validTime/gml:TimePeriod/gml:beginPosition",
        ns_seg,
    )

    endPosition = point.find(
        "aixm:timeSlice/aixm:RouteSegmentTimeSlice/gml:validTime/"
        "gml:TimePeriod/gml:endPosition",
        ns_seg,
    )

    upperLimit = point.find(
        "aixm:timeSlice/aixm:RouteSegmentTimeSlice/aixm:upperLimit",
        ns_seg,
    )

    lowerLimit = point.find(
        "aixm:timeSlice/aixm:RouteSegmentTimeSlice/aixm:lowerLimit",
        ns_seg,
    )

    routeFormed = point.find(
        "aixm:timeSlice/aixm:RouteSegmentTimeSlice/aixm:routeFormed",
        ns_seg,
    )

    start_designatedPoint = point.find(
        "aixm:timeSlice/aixm:RouteSegmentTimeSlice/aixm:start/"
        "aixm:EnRouteSegmentPoint/aixm:pointChoice_fixDesignatedPoint",
        ns_seg,
    )

    end_designatedPoint = point.find(
        "aixm:timeSlice/aixm:RouteSegmentTimeSlice/aixm:end/"
        "aixm:EnRouteSegmentPoint/aixm:pointChoice_fixDesignatedPoint",
        ns_seg,
    )

    start_navaid = point.find(
        "aixm:timeSlice/aixm:RouteSegmentTimeSlice/aixm:start/"
        "aixm:EnRouteSegmentPoint/aixm:pointChoice_navaidSystem",
        ns_seg,
    )

    end_navaid = point.find(
        "aixm:timeSlice/aixm:RouteSegmentTimeSlice/aixm:end/"
        "aixm:EnRouteSegmentPoint/aixm:pointChoice_navaidSystem",
        ns_seg,
    )

    directions = point.findall(
        "aixm:timeSlice/aixm:RouteSegmentTimeSlice/aixm:availability/"
        "aixm:RouteAvailability/aixm:direction",
        ns_seg,
    )

    if directions is None or len(directions) < 1:
        direction_str = "BOTH"
    elif directions is not None and len(directions) > 1:
        direction_str = directions[0].text
        for d in directions:
            if d.text != direction_str:
                direction_str = "BOTH"
    else:
        direction_str = directions[0].text

    # direction_str = direction.text if direction is not None else None

    beginPosition_str = (
        beginPosition.text if beginPosition is not None else None
    )

    endPosition_str = endPosition.text if endPosition is not None else None

    upperLimit_str = upperLimit.text if upperLimit is not None else None

    lowerLimit_str = lowerLimit.text if lowerLimit is not None else None

    routeFormed_str = (
        routeFormed.get("{http://www.w3.org/1999/xlink}href").split(":")[2]
        if routeFormed is not None
        else None
    )

    start_designatedPoint_str = (
        start_designatedPoint.get("{http://www.w3.org/1999/xlink}href").split(
            ":"
        )[2]
        if start_designatedPoint is not None
        else None
    )

    start_navaid_str = (
        start_navaid.get("{http://www.w3.org/1999/xlink}href").split(":")[2]
        if start_navaid is not None
        else None
    )

    end_designatedPoint_str = (
        end_designatedPoint.get("{http://www.w3.org/1999/xlink}href").split(
            ":"
        )[2]
        if end_designatedPoint is not None
        else None
    )

    end_navaid_str = (
        end_navaid.get("{http://www.w3.org/1999/xlink}href").split(":")[2]
        if end_navaid is not None
        else None
    )

    all_points_seg[identifier.text] = {
        "identifier": identifier.text,
        "beginPosition": beginPosition_str,
        "endPosition": endPosition_str,
        "upperLimit": upperLimit_str,
        "lowerLimit": lowerLimit_str,
        "start_designatedPoint": start_designatedPoint_str,
        "start_navaid": start_navaid_str,
        "end_designatedPoint": end_designatedPoint_str,
        "end_navaid": end_navaid_str,
        "routeFormed": routeFormed_str,
        "direction": direction_str,
    }

data_segments = pd.DataFrame.from_records(
    point for point in all_points_seg.values()
)

# %%
# MERGE START AND END NAVAIDS

start = pd.DataFrame(
    {
        "identifier": data_segments["identifier"],
        "start_pt": data_segments["start_designatedPoint"],
        "start_nav": data_segments["start_navaid"],
    }
)
start_merged = start["start_pt"].fillna(start["start_nav"])
data_segments["start_id"] = start_merged
data_segments = data_segments.drop(
    columns=["start_designatedPoint", "start_navaid"]
)

end = pd.DataFrame(
    {
        "identifier": data_segments["identifier"],
        "end_pt": data_segments["end_designatedPoint"],
        "end_nav": data_segments["end_navaid"],
    }
)
end_merged = end["end_pt"].fillna(end["end_nav"])
data_segments["end_id"] = end_merged
data_segments = data_segments.drop(
    columns=["end_designatedPoint", "end_navaid"]
)

# ADD NAVAID NAMES


to_merge_start = pd.DataFrame(
    {
        "start_id": aixm_navaids.data["id"],
        "start_name": aixm_navaids.data["name"],
    }
)
to_merge_end = pd.DataFrame(
    {"end_id": aixm_navaids.data["id"], "end_name": aixm_navaids.data["name"]}
)

data_segments = data_segments.merge(to_merge_start, on="start_id")
data_segments = data_segments.merge(to_merge_end, on="end_id")

# %%

routes = (
    pd.concat(
        [
            data_routes.query("prefix.notnull()").eval(
                "name = prefix + secondLetter + number"
            ),
            data_routes.query("prefix.isnull()").eval(
                "name =  secondLetter + number"
            ),
        ]
    )
    .drop(columns=["prefix", "secondLetter", "number"])
    .rename(columns={"identifier": "routeFormed"})
    .merge(data_segments, on="routeFormed")
)

# %%

# df = routes.query("name == 'UN869'")
cumul = []
for x, d in tqdm(df.groupby("routeFormed"), total=routes.routeFormed.nunique()):
    keys = dict(zip(d.start_id, d.end_id))
    start_set = set(d.start_id) - set(d.end_id)
    start = start_set.pop()
    unfold_list = [{"id": start}]
    while start is not None:
        start = keys.get(start, None)
        if start is not None:
            unfold_list.append({"id": start})
    cumul.append(
        pd.DataFrame.from_records(unfold_list)
        .merge(aixm_navaids.data)
        .assign(routeFormed=x, route=d.name.max())
        .drop(columns=["type", "description"])
    )

pd.concat(cumul)
# %%
airways_data = pd.concat(cumul)

# %%
airways_data
# %%
