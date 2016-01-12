"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 *
 * Problems - 1. Scalable matrix multiplication
 *            2. Using outer product for memory efficiency 
 *            3. Given matrix can have as many rows but 10 columns
 * 
 * ********************************************************************************************/ 
"""

from pyspark import SparkConf, SparkContext
import sys

''' sbaronia - this function adds two tuples
in the matrix RDD'''
def add_tuples(a,b):
	return tuple(sum(p) for p in zip(a,b))

'''sbaronia - this function takes line from the 
input file and does element wise multiplication 
to retrurn a 100 elements list which is equivalent
to 10x10 matrix'''
def element_wise_product(line):
	length = len(line)
	submatrix = []
	for i in range(length):
		for j in range(length):
			submatrix.append(float(line[i]) * float(line[j]))
	return submatrix

def main():
	inputs = sys.argv[1]
	output = sys.argv[2] 

	conf = SparkConf().setAppName('scalable multiplication')
	sc = SparkContext(conf=conf)
	assert sc.version >= '1.5.1'

	text = sc.textFile(inputs)

	# sbaronia - Split the row to get individual numbers
	row = text.map(lambda line: line.split())
	
	# sbaronia - calling element_wise_product on individual line 
	# and then adding all the returned 10x10 matrix to get
	# final matrix
	sub = row.map(element_wise_product).reduce(add_tuples)

	# sbaronia - writing formatted output to a file in 
	# a 10x10 matrix
	result = open(output, 'w')

	count = 0
	for i in range(len(sub)):
		result.write(str(sub[i]) + " ")
		count += 1
		if (count == 10):
			result.write("\n")
			count = 0

	result.close()

if __name__ == "__main__":
	main()

