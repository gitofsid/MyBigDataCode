"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * Topic - SPARK SQL
 * 
 * Problems -1. Read and store lines in better parquet format using Spark SQL
 *           2. Date should be changed in datetime object
 *           3. Create an RDD row object and output to parquet files
 *           4. Sum the bytes from every host
 * ********************************************************************************************/

"""


from pyspark import SparkConf, SparkContext
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import sys, json, re, datetime
from pyspark.sql import SQLContext, DataFrame, Row



def main():
	inputs = sys.argv[1] #input
	output = sys.argv[2] #output

	conf = SparkConf().setAppName('load logs')
	sc = SparkContext(conf=conf)
	assert sc.version >= '1.5.1'

	sqlContext = SQLContext(sc)
	text = sc.textFile(inputs)

	""" sbaronia - compiling the pattern for parsing the lines"""
	linere = re.compile("^(\\S+) - - \\[(\\S+) [+-]\\d+\\] \"[A-Z]+ (\\S+) HTTP/\\d\\.\\d\" \\d+ (\\d+)$")
	line_match = text.map(lambda line: linere.match(line)).filter(None)

	""" sbaronia - map the lines to create an RDD with names host, date, path, bytes in it """
	data = line_match.map(lambda line: Row(host=line.group(1),
									    date=datetime.datetime.strptime(line.group(2), '%d/%b/%Y:%H:%M:%S'),
									    path=line.group(3),
									    bytes=long(line.group(4))))

	""" sbaronia - create a dataframe from the rdd and write to parquet format"""
	data_df = sqlContext.createDataFrame(data).coalesce(1)
	data_df.write.format('parquet').save(output)

	""" sbaronia - read parquet files """
	parquet_df = sqlContext.read.parquet(output)
	parquet_df.show()

	""" sbaronia - group the table and then sum the bytes column 
	and show it to console """
	bytes_sum = parquet_df.groupBy().sum('bytes')
	bytes_sum.show()

if __name__ == "__main__":
	main()