"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * Topic - SPARK SQL
 *
 * Problems -1. Reuse reddit average code, use Spark SQL method
 *           2. Find average of every unique subreddit
 * ********************************************************************************************/

"""

from pyspark import SparkConf, SparkContext
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import sys, json
from pyspark.sql import SQLContext, DataFrame


def main():
	inputs = sys.argv[1]
	output = sys.argv[2]

	conf = SparkConf().setAppName('reddit average sql')
	sc = SparkContext(conf=conf)
	assert sc.version >= '1.5.1'

	sqlContext = SQLContext(sc)

	""" sbaronia - creating a schema with two fields 
	subreddit and score. False is not nullable """
	schema = StructType([StructField('subreddit', StringType(), False),
		                 StructField('score', IntegerType(), False)])
	
	""" sbaronia - read into dataframe with schema and in
	json forms"""
	comments = sqlContext.read.schema(schema).json(inputs)

	""" sbaronia - grouping by subreddit and calculating average 
	showing it on console """
	averages = comments.select('subreddit', 'score').groupby('subreddit').avg().coalesce(1).cache()
	averages.show()

	""" sbaronia - writing to a file """
	averages.write.save(output, format='json', mode='overwrite')

if __name__ == "__main__":
	main()