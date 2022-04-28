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
from traffic.core.spark import SparkTraffic
from pyspark.sql import functions as fn

data_path = "<fill here>"

#  df = spark.createDataFrame(quickstart.data)
#  spark_traffic = SparkTraffic(df)

spark_traffic = SparkTraffic.from_csv(data_path, spark)

st = (
    spark_traffic.withColumn("bla", fn.lit(1)).query(  # .assign(ILS="")
        "altitude < 3000"
    )  # Traffic -> None | Traffic -> THIS IS NOT ON THE STACK
    # Lazy iteration is triggered here by the .feature_lt method
    .feature_lt("vertical_rate_mean", -500)  # Flight -> bool
    #  .intersects(airports["LFPO"])            # Flight -> bool
    #  .next('aligned_on_ils("LFPO")')           # Flight -> bool
    .withColumn("blah", fn.lit(2))
    # .assign(airport="LFPO")
    # .assign(delta=lambda df: df.timestamp - df.timestamp.min())
    #  .last("10 min")                          # Flight -> None | Flight
    # Now evaluation is triggered on 4 cores
    # .eval(desc="landing at LFPO", spark=spark)
    # .eval(spark=spark)
)

st.sdf.show()
