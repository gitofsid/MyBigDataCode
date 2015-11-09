"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * 
 * Problems -1. Parse streaming data 
 *           
 * ********************************************************************************************/

"""


from pyspark import SparkConf, SparkContext
import sys, operator,re, string,json,unicodedata,datetime
from pyspark.streaming import StreamingContext
from operator import add


sc = SparkContext()
ssc = StreamingContext(sc, 1)
inputs = sys.argv[1]
output = sys.argv[2]

def add_tuples(a,b):
	return tuple(sum(p) for p in zip(a,b))

def func_for_every_rdd(rdd):
	if not rdd.isEmpty():
		tuple_sum = rdd.map(lambda (x,y): (x*x,x*y, x, y, 1)).reduce(add_tuples)
		count=tuple_sum[4]
		sum_y=tuple_sum[3]
		sum_x=tuple_sum[2]
		sum_xy=tuple_sum[1]
		sum_xx=tuple_sum[0]
		mean_xy=sum_xy/count
		mean_x=sum_x/count
		mean_y=sum_y/count
		mean_xx=sum_xx/count

		m=(mean_xy-(mean_x*mean_y))/(mean_xx-mean_x*mean_x)

		b=mean_y-m*mean_x

		save_obj = sc.parallelize([(m,b)], numSlices=1)
		save_obj.saveAsTextFile(output + '/' + datetime.datetime.now().isoformat().replace(':', '-'))

def main():

	ssc_input = ssc.socketTextStream("cmpt732.csil.sfu.ca", int(inputs))
	input_rdds=ssc_input.map(lambda var: map(float, var.split()))
	Result=input_rdds.foreachRDD(func_for_every_rdd)
	ssc.start()             
	ssc.awaitTermination(timeout=300) 


if __name__ == "__main__":
	main()
