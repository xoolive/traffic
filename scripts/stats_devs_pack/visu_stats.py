# %%

import altair as alt
import pandas as pd


# %%

test = pd.read_parquet("../t2_0722_TEST.parquet")
testday = pd.read_parquet("../t2_140722.parquet")
# %%
# stats_2 has a maximum forward time instead of a min
# (prediction's max length is 15 minutes)
# stats_4 utilise une prediction de 15 minutes (sinon, risque d'être trop court)
# stats_5, on ne garde que les trous de + de 2 min contenus dans BDX
# stats_6 with min distance (in aligned_on_navaid) = 120
# stats_7 with same time for pred and trou
# stats_8 angle_precision = 2 and prediction with fp included
# stats_14 with a revised stop_voisins reduced if altitude drops

stats_9 = pd.read_parquet("stats_9.parquet")
stats_13 = pd.read_parquet("stats_13.parquet")
stats_14 = pd.read_parquet("stats_14.parquet")
stats_day = pd.read_parquet("stats_12.parquet")
# stats = stats.drop(columns=["Unnamed: 0"])

# %%

# para trucs


stats = pd.read_parquet("para")
# %%


# %%


stats["duration"] = stats["duration"].dt.total_seconds()
stats["min_f_dev"] = stats["min_f_dev"].dt.total_seconds()
stats["difference"] = stats["min_f_dist"] - stats["min_fp_dist"]
stats = stats[stats.min_f_dist != 0.0]
stats["delay"] = (stats["min_fp_time"] - stats["start"]).dt.total_seconds()

# %%
# eliminate cases where 1 flight -> 2 flight_ids (distance between flights=0)
# we lose the info (temporary)


# %% LAYERED CHART ALTAIR

source = stats[
    ["min_fp_dist", "min_f_dist"]
]  # .query('min_fp_dist<50 and min_f_dist<50')
source = source.dropna()
chart1 = (
    alt.Chart(source)
    .transform_fold(
        ["min_fp_dist", "min_f_dist"], as_=["Experiment", "Measurement"]
    )
    .transform_calculate(
        Experiment_Label="datum.Experiment == 'min_fp_dist' ? 'Predicted' : 'Actual'"
    )
    .mark_area(opacity=0.3, interpolate="step", binSpacing=0)
    .encode(
        alt.X("Measurement:Q", title="Separation (NM)").bin(maxbins=100),
        alt.Y("count()", title="Count").stack(None),
        alt.Color("Experiment_Label:N", title="Values"),
    )
    .properties(height=150)
    # .properties(
    #     title="Distribution of actual and predicted separation"  # Add the title here
    # )
)

chart2 = (
    alt.Chart(source.query("min_fp_dist<40 and min_f_dist<40"))
    .transform_fold(
        ["min_fp_dist", "min_f_dist"], as_=["Experiment", "Measurement"]
    )
    .transform_calculate(
        Experiment_Label="datum.Experiment == 'min_fp_dist' ? 'Predicted' : 'Actual'"
    )
    .mark_area(opacity=0.3, interpolate="step", binSpacing=0)
    .encode(
        alt.X("Measurement:Q", title="Separation (NM)").bin(maxbins=100),
        alt.Y("count()", title="Count").stack(None),
        alt.Color("Experiment_Label:N", title="Values"),
    )
    .properties(height=150)
)

alt.hconcat(chart1, chart2)  # .properties(
# title="Distribution of actual and predicted separation",  # height=200,
# )
# chart.save("distrib_sep4.pdf")

# %%
# DELAYS IN WHOLE DATASET VS ONLY FP < 5

all = (
    alt.Chart(stats)
    .mark_bar(color="teal")
    .encode(
        alt.X(
            "delay", title="delay in seconds", bin=alt.BinParams(maxbins=100)
        ),
        # alt.X("delay", title="delay"),
        alt.Y("count()", title="count"),
        # alt.Y("count()", title="number of trajectories"),
        # alt.Color("status", scale=alt.Scale(range=range_)),
    )
    .properties(height=200, width=500, title="Delay in whole dataset")
)

inf5 = (
    alt.Chart(stats.query("min_fp_dist<5"))
    .mark_bar(color="teal")
    .encode(
        alt.X(
            "delay", title="delay in seconds", bin=alt.BinParams(maxbins=100)
        ),
        # alt.X("delay", title="delay"),
        alt.Y("count()", title="count"),
        # alt.Y("count()", title="number of trajectories"),
        # alt.Color("status", scale=alt.Scale(range=range_)),
    )
    .properties(height=200, width=500, title="In cases where min_fp_dist<5")
)

(all & inf5).resolve_scale(x="shared")

# %%

# DIFFERENCE IN WHOLE DATASET VS ONLY FP < 5

all = (
    alt.Chart(stats)
    .mark_bar(color="teal")
    .encode(
        alt.X("difference", title="difference", bin=alt.BinParams(maxbins=100)),
        # alt.X("delay", title="delay"),
        alt.Y("count()", title="count"),
        # alt.Y("count()", title="number of trajectories"),
        # alt.Color("status", scale=alt.Scale(range=range_)),
    )
    .properties(height=200, width=500, title="Difference in whole dataset")
)

inf5 = (
    alt.Chart(stats.query("min_fp_dist<5"))
    .mark_bar(color="teal")
    .encode(
        alt.X("difference", title="difference", bin=alt.BinParams(maxbins=100)),
        # alt.X("delay", title="delay"),
        alt.Y("count()", title="count"),
        # alt.Y("count()", title="number of trajectories"),
        # alt.Color("status", scale=alt.Scale(range=range_)),
    )
    .properties(height=200, width=500, title="In cases where min_fp_dist<5")
)

(all & inf5).resolve_scale(x="shared")

# %%

# DELAY ACCORDING TO DIFFERENCE

(
    alt.Chart(stats.query("min_fp_dist<min_f_dist and min_fp_dist<10"))
    .mark_bar(color="teal")
    .encode(
        alt.X(
            "delay", title="delay", bin=alt.BinParams(maxbins=100)
        ),  # , bin=alt.BinParams(maxbins=100)),
        # alt.X("delay", title="delay"),
        alt.Y("difference", title="difference"),
        # alt.Y("count()", title="number of trajectories"),
        # alt.Color("status", scale=alt.Scale(range=range_)),
    )
    .properties(height=200, width=500, title="test")
)

# %%

# DIFFERENCE ACCORDING TO FP DIST FOR FP DIST <40

(
    alt.Chart(stats.query("min_fp_dist<min_f_dist and min_fp_dist<10"))
    .mark_bar(color="teal")
    .encode(
        alt.X("min_fp_dist", title="min_fp_dist"),
        alt.Y("difference", title="difference"),
    )
    .properties(height=200, width=500, title="fp_dist/difference")
)

# %%
# FLOWN / FP

base = (
    alt.Chart(stats.query("min_fp_dist<30"))
    .mark_line(color="teal")
    .encode(
        alt.X("min_fp_dist", title="min_fp_dist"),
        alt.Y("min_f_dist", title="min_f_dist"),
    )
    .properties(height=200, width=500, title="fp_dist/f_dist")
)

five_line_h = (
    alt.Chart(pd.DataFrame({"min_f_dist": [5], "color": ["red"]}))
    .mark_rule()
    .encode(alt.Y("min_f_dist"), color=alt.Color("color:N", scale=None))
)

five_line_v = (
    alt.Chart(pd.DataFrame({"min_fp_dist": [5], "color": ["red"]}))
    .mark_rule()
    .encode(alt.X("min_fp_dist"), color=alt.Color("color:N", scale=None))
)

final_chart = alt.layer(five_line_h, five_line_v, base)

final_chart
# %%

# essayer avec le rapport duration_deviation/duration_trajectory

# base = (
#     alt.Chart(stats)
#     .mark_bar()
#     .encode(
#         alt.X("duration_m:Q", title="duration"),
#         alt.Y("count()", title="number of trajectories"),
#         # alt.Color("status", scale=alt.Scale(range=range_)),
#     )
#     .properties(
#         height=150, width=600, title="Unaligned time in all trajectories"
#     )
#     .transform_calculate(duration_m="datum.duration/60")
# )

# (base & base.transform_filter("datum.duration_m > 2")).resolve_scale(x="shared")


# %%

# stats.query("4*60 < duration < 5*60")
# %%
# alt.Chart(stats.query("min_fp_dist<10 and min_fp_dist<min_f_dist")).mark_line(
chart = (
    alt.Chart(stats.query("min_f_dist<40"))
    .mark_bar(color="teal")
    .encode(
        alt.X("min_f_dist", title="min f", bin=alt.BinParams(maxbins=100)),
        alt.Y("count()", title="number of trajectories"),
        # alt.Color("status", scale=alt.Scale(range=range_)),
    )
    .properties(height=300, title="Minimum distance")
)
vertical_line = (
    alt.Chart(pd.DataFrame({"min_f_dist": [5], "color": ["red"]}))
    .mark_rule()
    .encode(alt.X("min_f_dist"), color=alt.Color("color:N", scale=None))
)
chart + vertical_line

# %%
alt.Chart(stats.query("min_fp_dist<min_f_dist and min_fp_dist<10")).mark_bar(
    color="teal"
).encode(
    # alt.X("min_fp_dist", title="min_fp_dist", bin=alt.BinParams(maxbins=100)),
    alt.X("nb_voisins", title="nb_voisins"),  # bin=alt.BinParams(maxbins=50)),
    alt.Y("difference", title="difference"),
    # alt.Y("count()", title="number of trajectories"),
    # alt.Color("status", scale=alt.Scale(range=range_)),
).properties(
    height=300, title="Minimum predicted distance"
)
# %%
alt.Chart(stats.query("min_f_dist<50")).mark_bar(color="teal").encode(
    alt.X("min_f_dist", title="min_f_dist", bin=alt.Bin(maxbins=100)),
    alt.Y("min_fp_dist", title="min_fp_dist"),
    # alt.Color("status", scale=alt.Scale(range=range_)),
).properties(height=300, title="Minimum predicted distance")
# %%
alt.Chart(stats.query("min_fp_dist<min_f_dist and min_fp_dist<10")).mark_line(
    color="teal"
).encode(
    alt.X("min_fp_dist", title="min_fp_dist", bin=alt.BinParams(maxbins=100)),
    alt.Y("difference", title="difference"),
    # alt.Color("status", scale=alt.Scale(range=range_)),
).properties(
    height=300, title="Minimum predicted distance"
)

# %%


chart = (
    alt.Chart(stats.query("min_f_dist<5"))
    .mark_bar(color="#9ba4ff")
    .encode(
        alt.X("min_f_dist", title="min_f_dist", bin=alt.BinParams(maxbins=100)),
        alt.Y("count()", title="count"),
    )
    .properties(height=300, title="Difference between flown and predicted")
)

# Create a vertical line at 0
zero_line = (
    alt.Chart(pd.DataFrame({"difference": [0], "color": ["red"]}))
    .mark_rule(text="prout")
    .encode(alt.X("difference"), color=alt.Color("color:N", scale=None))
)
eight_line = (
    alt.Chart(pd.DataFrame({"difference": [8.3], "color": ["green"]}))
    .mark_rule(text="prout")
    .encode(alt.X("difference"), color=alt.Color("color:N", scale=None))
)

# ten_line = (
#     alt.Chart(pd.DataFrame({"difference": [8.3], "color": ["green"]}))
#     .mark_rule(text="prout")
#     .encode(alt.X("difference"), color=alt.Color("color:N", scale=None))
# )

three_line = (
    alt.Chart(pd.DataFrame({"difference": [2.5], "color": ["magenta"]}))
    .mark_rule(text="prout")
    .encode(alt.X("difference"), color=alt.Color("color:N", scale=None))
)

# Layer the chart and the vertical line
final_chart = alt.layer(chart, zero_line, eight_line, three_line)

final_chart


# %%


alt.Chart(stats).mark_circle().encode(
    alt.X("delay", title="delay"),  # , bin=alt.BinParams(maxbins=100)),
    alt.Y("max_dev_angle", title="angle"),
).properties(height=300, title="test")


# %%
import altair as alt

# Your existing chart code
chart = (
    alt.Chart(stats.query("min_f_dist<50"))
    .mark_circle()
    .encode(
        alt.X("min_f_dist", title="min_f_dist"),
        alt.Y("min_fp_dist", title="min_fp_dist"),
    )
    .properties(height=300, title="test")
)

# Add a line for min_f_dist = min_fp_dist
line = (
    alt.Chart(pd.DataFrame({"x": [0, 50], "y": [0, 50]}))
    .mark_rule(color="red")
    .encode(x="y", y="x")
)

# Combine the scatter plot and the line
combined_chart = chart + line

# Show the combined chart
combined_chart


# %%
alt.Chart(stats.query("min_fp_dist<min_f_dist")).mark_bar().encode(
    alt.X(
        "min_fp_dist", title="min_fp_dist"
    ),  # , bin=alt.BinParams(maxbins=100)),
    alt.Y("difference", title="difference"),
    # alt.Color("status", scale=alt.Scale(range=range_)),
).properties(height=300, title="duration")

# %%
import altair as alt


chart1 = (
    alt.Chart(stats.query("min_f_dist<50"))
    .mark_bar(color="#4c78a8")
    .encode(
        alt.X("min_f_dist", title="min_f_dist", bin=alt.BinParams(maxbins=100)),
        alt.Y("count()", title="count"),
    )
    .properties(height=150, title="Flown")
)

chart2 = (
    alt.Chart(stats.query("min_f_dist<50"))
    .mark_bar(color="#4c78a8")
    .encode(
        alt.X(
            "min_fp_dist", title="min_fp_dist", bin=alt.BinParams(maxbins=100)
        ),
        alt.Y("count()", title="count"),
    )
    .properties(height=150, title="Predicted")
)

# Vertically concatenate the two charts
combined_chart = alt.vconcat(chart1, chart2)

# Display the combined chart
combined_chart  # .resolve_scale(x="shared")


# %%
import altair as alt


# %%  TESTS AVEC RICHARD
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np

df = stats.query("min_f_dist<15 and min_fp_dist<15")
x = df.min_fp_dist.values
# x = df.difference.values
y = df.min_f_dist.values - df.min_fp_dist.values  # df.difference.values
plt.scatter(x, y, alpha=0.1)
# plt.plot([0, 50], [0, 0], c="red")
plt.scatter(x, 5 - x, alpha=0.1)
model = MedianKNNRegressor(n_neighbors=50)
xinp = np.expand_dims(x, 1)
model.fit(xinp, y)
plt.scatter(x, model.predict(xinp), c="pink", alpha=0.1)
# plt.plot([10,10], [0, 20], c="green")
# plt.vlines(x=8, ymin=-20, ymax=20, color="red", linestyles="dashed")


# %%
from sklearn.utils import check_array
from sklearn.neighbors._base import _get_weights


class MedianKNNRegressor(KNeighborsRegressor):
    def __init__(self, quantile, **kwargs):
        super().__init__(**kwargs)
        self.quantile = quantile

    def predict(self, X):
        X = check_array(X, accept_sparse="csr")

        neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        ######## Begin modification
        if weights is None:
            y_pred = np.quantile(_y[neigh_ind], q=self.quantile, axis=1)
        else:
            # y_pred = weighted_median(_y[neigh_ind], weights, axis=1)
            raise NotImplementedError("weighted median")
        ######### End modification

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred


# %%
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np

df = stats.query("min_f_dist<50 and min_fp_dist<50")
x = df.min_fp_dist.values

y = df.difference.values
plt.axhline(0, color="gray", linestyle="-", linewidth=0.5)
# y = df.min_f_dist.values  # -df.min_fp_dist.values#df.difference.values
plt.scatter(x, y, alpha=0.1)
# plt.plot([0, 50], [0, 0], c="red")
# plt.scatter(x, 5 - x, alpha=0.1)
xinp = np.expand_dims(x, 1)


def draw(q):
    model = MedianKNNRegressor(n_neighbors=100, weights="uniform", quantile=q)
    model.fit(xinp, y)
    return model


# z = zip(x, draw(0.5).predict(xinp))
# z = sorted(z)
# Combine a and b and sort by the elements of a
z = sorted(zip(x, draw(0.5).predict(xinp)), key=lambda x: x[0])
sorted_x, sorted_draw = zip(*z)

# plt.scatter(x, draw(0.2).predict(xinp), c="red", alpha=0.1)
plt.plot(sorted_x, sorted_draw, c="black", alpha=1, linewidth=3.5)

plt.xlabel("Min_pred [NM]")
plt.ylabel("Diff [NM]")
plt.title("")
# plt.scatter(x, draw(0.8).predict(xinp), c="green", alpha=0.1)
# plt.plot([10,10], [0, 20], c="green")
# plt.grid(axis='y', linewidth=0.5)
plt.vlines(x=8, ymin=y.min(), ymax=y.max(), color="red", linestyles="dashed")
# plt.hlines(y=0, xmin=x.min(), xmax=x.max(), color="red", linestyles="dashed")

# %%

# test d'affichage en altair
import altair as alt
import pandas as pd
from IPython.display import display

# Your data
data = pd.DataFrame(
    {"x": x, "y": y, "sorted_x": sorted_x, "sorted_draw": sorted_draw}
)

# Create the scatter plot
scatter_plot = (
    alt.Chart(data)
    .mark_circle(opacity=0.1)
    .encode(
        x=alt.X(
            "x",
            title="Predicted min separation",
            scale=alt.Scale(domain=[0, 50]),
        ),
        y=alt.Y(
            "y",
            title="Predicted - Actual min separation",
            scale=alt.Scale(domain=[-15, 15]),
        ),
    )
)

# Create the line plot
line_plot = (
    alt.Chart(data)
    .mark_line(color="black", strokeWidth=3.5)
    .encode(
        x=alt.X(
            "sorted_x",
            title="Predicted min separation",
            scale=alt.Scale(domain=[0, 50]),
        ),
        y=alt.Y(
            "sorted_draw",
            title="Predicted - Actual min separation",
            scale=alt.Scale(domain=[-15, 15]),
        ),
    )
)

# Create the vertical dashed line
vline = (
    alt.Chart(pd.DataFrame({"vline_x": [8]}))
    .mark_rule(color="red", strokeDash=[3, 3])
    .encode(
        x=alt.X("vline_x", title="Threshold", scale=alt.Scale(domain=[0, 50])),
        y=alt.Y(
            "min(y):Q",
            title="Predicted - Actual min separation",
            scale=alt.Scale(domain=[0, 15]),
        ),
    )
)

# Combine all the charts
final_chart = scatter_plot + line_plot + vline


final_chart.configure_title(fontSize=15).properties(
    width=800, height=200, title="Predicted vs. Actual Min Separation"
)

# Display the chart in the Jupyter Notebook
display(final_chart)

# %%
