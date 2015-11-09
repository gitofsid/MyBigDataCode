"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * 
 * Problems -1. Create table with data, station and maximum temperature difference
 *              recorded for that date  
 *           2. Clean the output by printing without comma, in readable form
 * ********************************************************************************************/

"""

from pyspark import SparkConf, SparkContext
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import sys
from pyspark.sql import SQLContext, DataFrame


def main():
	inputs = sys.argv[1]
	output = sys.argv[2]

	conf = SparkConf().setAppName('temp range')
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

	
	df_min = df.filter(df.Observation == 'TMIN').withColumnRenamed('Value', 'TMIN_VAL')
	df_max = df.filter(df.Observation == 'TMAX').withColumnRenamed('Value', 'TMAX_VAL')
	

	cond = ['Station', 'Date', 'Mflag','Qflag']
	join_int = df_max.join(df_min, cond, 'inner')
	join_qflag = join_int.filter(join_int.Qflag == '')
	
	join_df = join_qflag.select('Station', 'Date', df_max.TMAX_VAL - df_min.TMIN_VAL) \
					.withColumnRenamed('(TMAX_VAL - TMIN_VAL)', 'Range').cache()

	by_date = join_df.select('Date','Range') \
					 .groupby('Date') \
					 .max('Range') \
					 .withColumnRenamed('max(Range)', 'Range').cache()

	final = join_df.join(by_date, ['Date', 'Range'], 'inner').dropDuplicates()

	final_rdd = final.rdd.map(lambda x: x.Date + ' ' + x.Station + ' ' + str(x.Range))
	final_rdd.repartition(1).saveAsTextFile(output)


if __name__ == "__main__":
	main()