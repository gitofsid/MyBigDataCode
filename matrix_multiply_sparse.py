"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * 
 * Problems - 1. Scalable multiplication using sparse csr_matrix
 *            2. Use csr_matrix and its helper functions for multiplication purposes
 *            3. Output in the form of non-zero index:data, same as input
 * 
 * ********************************************************************************************/ 
"""

from pyspark import SparkConf, SparkContext
from scipy.sparse import *
from scipy import *
import sys

'''sbaronia - this function converts a row from input file
to csr_matrix format. The input is in nonzero-index:data
format for every line'''
def convert_to_csr_matrix(line):
	indices_list = []
	indptr_list = [0]
	data_list = []
	length = len(line)
	indptr_list.append(length)
	for i in range(length):
		x,y = line[i].split(':')
		indices_list.append(int(x))
		data_list.append(float(y))

	'''sbaronia - make csr row matrix and then make a column matrix using transpose
	helper function and then multiply these to find 100x100 matrix for each row'''
	cmatrix_row = csr_matrix( (array(data_list), array(indices_list), array(indptr_list)),
						shape=(1,100))
	cmatrix_column = cmatrix_row.transpose()

	submatrix = cmatrix_column.multiply(cmatrix_row).todense()

	return submatrix

def main():
	inputs = sys.argv[1]
	output = sys.argv[2] 

	conf = SparkConf().setAppName('sparse scalable multiplication')
	sc = SparkContext(conf=conf)
	assert sc.version >= '1.5.1'

	text = sc.textFile(inputs)
	row = text.map(lambda line: line.split())

	'''sbaronia - convert each line of input to csr matrix and then 
	add all to find final matrix'''
	sub = row.map(convert_to_csr_matrix).reduce(lambda a,b: a+b)

	'''sbaronia - convert final matrix to a list so it can be
	saved to a file with format'''
	sub_list = sub.tolist()

	# sbaronia - writing formatted output to a file in 
	# index:data form for non-zero element
	result = open(output, 'w')

	for i in range(len(sub_list)):
		for j in range(len(sub_list)):
			# sbaronia - we dont use zero elements
			if(sub_list[i][j] != 0.0):
				result.write(str(j) + ':' + str(sub_list[i][j]) + " ")
			if (j == 99):
				result.write("\n")

	result.close()

if __name__ == "__main__":
	main()

