# %%

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

conf = SparkConf()
conf.setAppName("my_spark_app")
conf.setMaster("spark://160.85.67.62:7077")
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
