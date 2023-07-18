# %%

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

conf = SparkConf()
conf.setAppName("my_spark_app")
# conf.setMaster("spark://160.85.67.62:7077")
conf.set("spark.executor.memory", "600G")
conf.set("spark.driver.memory", "12G")
conf.set("spark.driver.maxResultSize", "12G")
conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
conf.set("spark.sql.session.timeZone", "UTC")

spark = SparkSession.builder.config(conf=conf).getOrCreate()

# %%
from traffic.data.samples import quickstart
from traffic.data import airports

quickstart = quickstart.assign_id().eval()

t = (
    quickstart.assign(ILS="")
    .iterate_lazy()
    .query(
        "altitude < 3000"
    )  # Traffic -> None | Traffic -> THIS IS NOT ON THE STACK
    # Lazy iteration is triggered here by the .feature_lt method
    .feature_lt("vertical_rate_mean", -500)  # Flight -> bool
    .intersects(airports["LFPO"])  # Flight -> bool
    .next('aligned_on_ils("LFPO")')  # Flight -> bool
    # .assign(airport="LFPO")
    # .assign(delta=lambda df: df.timestamp - df.timestamp.min())
    .last("10 min")  # Flight -> None | Flight
    # Now evaluation is triggered on 4 cores
    # .eval(desc="landing at LFPO", spark=spark)
    .eval(spark=spark)
)
t

# %%

from traffic.core.spark import SparkTraffic

sample = SparkTraffic.from_parquet(
    "/home/xo/sample_traffic/sample_traffic_12_00.parquet",
    spark=spark,
)
# %%
sample.show()
# %%
from traffic.data import airports

ts = (
    sample.resample("1s")
    .intersects(airports["LSZH"])
    .has('aligned_on_ils("LSZH")')
    # .traffic
)
# ts.explain()
t = ts.traffic
# %%
from ipyleaflet import Map

m = Map(center=airports["LSZH"].latlon, zoom=9)
for flight in t:
    m.add(flight)
m

# %%
ts = (
    sample.inside_bbox((-10, 30, 30, 60))
    .resample("1s")
    .has("holding_pattern")
    # .traffic
)
# ts.explain()
t = ts.traffic
t
# %%
from ipyleaflet import Map

m = Map(center=(45, 10), zoom=4)
for flight in t[:30]:
    m.add(flight, highlight=dict(red="holding_pattern"))
m

# %%
