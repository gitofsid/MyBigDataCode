"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * Problems -1. Euler constant estimate using Spark
 *           
 * ********************************************************************************************/

"""

from pyspark import SparkConf, SparkContext
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql import SQLContext, DataFrame
import sys, string, random


""" sbaronia - this function finds final
count per iteration"""
def find_sum_iterations(iterations):
	count = 0
	for j in xrange(iterations):
		sum_itr = 0.0
		while sum_itr < 1 :
			sum_itr = sum_itr + random.random()
			count = count + 1
	return count


def main():
	inputs_number = sys.argv[1]
	conf = SparkConf().setAppName('euler estimation')
	sc = SparkContext(conf=conf)
	assert sc.version >= '1.5.1'
	
	"""sbaronia - number of slices by which we will
	make partitions and create an rdd."""
	slices = 1000
	num_itr = int(inputs_number)/slices
	iter_list = []
	for i in xrange(slices):
		iter_list.append(num_itr)

	iter_rdd = sc.parallelize(iter_list, numSlices=slices)

	"""sbaronia - call the function per value of slice and sum the 
	count everytime"""
	f_count = iter_rdd.map(lambda line : find_sum_iterations(int(line))).sum()
	count_iteration = sc.parallelize([f_count, int(inputs_number)])

	print("[Count, Iterations]")
	print(count_iteration.collect())

	
if __name__ == "__main__":
	main()