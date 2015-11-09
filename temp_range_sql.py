"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * 
 * Problems -1. Same logic of temp_range.py but with SQL and tables
 *           
 * ********************************************************************************************/

"""


from pyspark import SparkConf, SparkContext
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import sys
from pyspark.sql import SQLContext, DataFrame


def main():
	inputs = sys.argv[1]
	output = sys.argv[2]

	conf = SparkConf().setAppName('temp range sql')
	sc = SparkContext(conf=conf)
	assert sc.version >= '1.5.1'

	text = sc.textFile(inputs)
	sqlContext = SQLContext(sc)

	schema = StructType([StructField('Station', StringType(), True),
						StructField('Date', StringType(), True),
						StructField('Observation', StringType(), True),
						StructField('Value', IntegerType(), True),
						StructField('Mflag', StringType(), True),	
						StructField('Qflag', StringType(), True)])


	df = sqlContext.read \
			 .format('com.databricks.spark.csv') \
			 .schema(schema) \
			 .load(inputs).cache()

	
	df_min = df.filter(df.Observation == 'TMIN')
	df_max = df.filter(df.Observation == 'TMAX')
	
	df_min.registerTempTable('df_min')
	df_max.registerTempTable('df_max')

	join_qflag = sqlContext.sql("""
	SELECT 	df_max.Station, df_max.Date,
			df_max.Value,df_min.Value,
			df_max.Value-df_min.Value Range
	FROM 	df_max,df_min
	where	df_max.Station = df_min.Station and 
			df_max.Date = df_min.Date and 
			df_max.Qflag = df_min.Qflag and 
			df_min.Qflag = ''
	""")

	join_qflag.registerTempTable('join_qflag')

	by_date = sqlContext.sql("""
	SELECT	Date as Date_1, max(Range) as Range_1
	FROM 	join_qflag 
	GROUP BY Date
	""")
	by_date.registerTempTable('by_date')

	final = sqlContext.sql("""
	SELECT 	join_qflag.Date, join_qflag.Station, join_qflag.Range
	FROM   	join_qflag,by_date
	where  	join_qflag.Date = by_date.Date_1 and
			join_qflag.Range = by_date.Range_1
	""")
	
	final_rdd = final.rdd.map(lambda x: x.Date + ' ' + x.Station + ' ' + str(x.Range))
	final_rdd.repartition(1).saveAsTextFile(output)


if __name__ == "__main__":
	main()