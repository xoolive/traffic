# Simple test to check if your containers setup with docker-compose are working
# properly.
# First, run the following command on the command line in the root directory of
# the repository:
# 
#  docker compose -f docker-compose-spark.yml up
#
# This will start up the three containers, one Spark master node, one Spark
# worker node, and a container with Python, traffic and pyspark installed.
# You can then either access the jupyter notebook server via your browser or you
# can connect your VSCode and run this script directly in it (or do whatever you
# actually want to work on).
# Once you're done, you can clean everything up with the command
#
# docker compose -f docker-compose-spark.yml down
#


from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SimpleApp").master("spark://localhost:7077").getOrCreate()
sc = spark.sparkContext

rdd = sc.parallelize(range(100 + 1))
rdd.sum()