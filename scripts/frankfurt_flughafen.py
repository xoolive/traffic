# %%
import altair as alt
from traffic.data import airports

airport = airports["EDDF"]
assert airport is not None

airport_layout = airport._openstreetmap().data

subset = airport_layout.query(
    'aeroway in ["terminal", "runway", "taxiway"] and '
    '(aeroway != "taxiway") or (ref.notnull() and ref.str[0] in ["L", "N"])'
)

gates = (
    airport_layout.query(
        'aeroway == "gate" and ref.notnull() '
        'and ref.str[0] in ["A", "B", "C", "D", "E"] '
        "and ref.str.len() <= 3"
    )
    .sort_values("ref")
    .loc[::5, ["latitude", "longitude", "ref"]]
)

terminal = subset.query('aeroway == "terminal"').unary_union
boundaries = terminal.minimum_rotated_rectangle.buffer(0.007)

subset = subset.loc[subset.intersects(boundaries)]
subset = subset.assign(geometry=subset.intersection(boundaries))

base = alt.Chart(subset)

text_channels = (
    alt.Latitude("latitude:Q"),
    alt.Longitude("longitude:Q"),
    alt.Text("ref:N"),
)

gate_mark = alt.Chart(gates).encode(*text_channels)

chart = (
    alt.layer(
        alt.Chart(terminal).mark_geoshape(stroke=None, color="#4c78a8"),
        base.mark_geoshape(
            strokeWidth=5, filled=False, color="#bab0ac"
        ).transform_filter("datum.aeroway == 'taxiway'"),
        base.mark_geoshape(
            strokeWidth=20, filled=False, color="#79706e"
        ).transform_filter("datum.aeroway == 'runway'"),
        base.mark_text(angle=341, color="white", fontSize=14)
        .transform_filter("datum.aeroway == 'runway'")
        .encode(*text_channels),
        gate_mark.mark_text(
            color="white",
            font="Ubuntu",
            fontSize=14,
            fontWeight="bold",
        ),
        gate_mark.mark_text(
            color="black",
            font="Ubuntu",
            fontSize=14,
            fontWeight=400,
        ),
    )
    .configure_view(stroke=None)
    .properties(width=800, height=400)
)

chart

# %%
