
"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 *
 * Problems - 1. Using output files from Q1 implement LinearRegressionWithSGD and find RMSE
 * 
 * ********************************************************************************************/ 
"""

import sys,string,math,random
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, DataFrame
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.feature import Normalizer

'''sbaronia - run LinearRegressionWithSGD on training and testing
and return root mean square error'''
def regression_and_error(train_x,test_x,step_x):
	model = LinearRegressionWithSGD.train(train_x, iterations=500, step=step_x)
	valpred = test_x.map(lambda test: (test.label, model.predict(test.features))).cache()
	
	rmse = math.sqrt(valpred.map(lambda (v, p): (v-p)**2).reduce(lambda x, y: x + y)/valpred.count())	
	return rmse

'''sbaroina - return labeled points with rating and tf-idf vector'''
def unormalized_labeledpoint(line):
	line_spl = line.split(' :: ')
	return LabeledPoint(line_spl[0], SparseVector.parse(line_spl[1]))

'''sbaronia - return normalized labeled points with rating and tf-idf vector'''
def normalized_labeledpoint(line,nor):
	line_spl = line.split(' :: ')
	return LabeledPoint(line_spl[0], nor.transform(SparseVector.parse(line_spl[1])))

'''sbaronia - this function does validation by taking training data
and splitting it into 70:30 ratio and runs regression with different
step size to find the best step size giving least RMSE'''
def validation(training_labeledpoint):
	step_best = 0.0
	rmse_best = 0.0
	for i in range(5):		
		
		val_train, val_test = training_labeledpoint.randomSplit([0.7, 0.3], seed=0)
		step = float(i+1)/100
		rmse = regression_and_error(val_train,val_test,step)
		
		if i == 0:
			rmse_best = rmse

		if rmse <= rmse_best:
			rmse_best = rmse
			step_best = step

		print("RMSE = " + str(rmse) + " Step = " + str(step) + " RMSE Best = " + str(rmse_best) + " Step best = " + str(step_best))

	return step_best

def main():
	input_train = sys.argv[1]
	input_test = sys.argv[2]
	
	conf = SparkConf().setAppName('TF-IDF Linear Regression Model')
	sc = SparkContext(conf=conf)
	assert sc.version >= '1.5.1'

	'''sbaronia - get training and testing rdds from
	files saved in question 1.'''
	train = sc.textFile(input_train).cache()
	test = sc.textFile(input_test).cache()

	'''sbaronia - get unnormalized training and testing data'''
	unnorm_train = train.map(unormalized_labeledpoint).cache()

	unnorm_test = test.map(unormalized_labeledpoint).cache()

	'''sbaronia - find L2 normalized training and testing data'''
	nor = Normalizer(2)

	norm_train = train.map(lambda line: normalized_labeledpoint(line,nor)).cache()

	norm_test = test.map(lambda line: normalized_labeledpoint(line,nor)).cache()

	'''sbaronia - find best step size for unnormalized data
	and report best RMSE'''
	step_best_nonorm = validation(unnorm_train)

	RMSE_nonorm = regression_and_error(unnorm_train,unnorm_test,step_best_nonorm)

	print("Final RMSE(No Normalization) = " + str(RMSE_nonorm) + "  Best Step size = " + str(step_best_nonorm))

	'''sbaronia - find best step size for normalized data
	and report best RMSE'''

	step_best_norm = validation(norm_train)

	RMSE_norm = regression_and_error(norm_train,norm_test,step_best_norm)

	print("Final RMSE(Normalization) = " + str(RMSE_norm) + "  Best Step size = " + str(step_best_norm))


if __name__ == "__main__":
	main()

'''
Answers - On reviews_Pet_Supplies_p1.json data set

To run on local-

spark-submit --master local[*] tf_idf_amazon_q1.py output-p1/train-text/part-00000 output-p1/test-text/part-00000

Output -

Without Normalization:

RMSE = 2.89040472822 Step = 0.01 RMSE Best = 2.89040472822 Step best = 0.01
RMSE = 2.73712591423 Step = 0.02 RMSE Best = 2.73712591423 Step best = 0.02
RMSE = 2.67871385171 Step = 0.03 RMSE Best = 2.67871385171 Step best = 0.03
RMSE = 2.64640232335 Step = 0.04 RMSE Best = 2.64640232335 Step best = 0.04
RMSE = 2.62617449726 Step = 0.05 RMSE Best = 2.62617449726 Step best = 0.05
Final RMSE(No Normalization) = 2.63965200335  Best Step size = 0.05

With Normalization:

RMSE = 4.30550043639 Step = 0.01 RMSE Best = 4.30550043639 Step best = 0.01
RMSE = 4.29097973223 Step = 0.02 RMSE Best = 4.29097973223 Step best = 0.02
RMSE = 4.26687264888 Step = 0.03 RMSE Best = 4.26687264888 Step best = 0.03
RMSE = 4.23376384531 Step = 0.04 RMSE Best = 4.23376384531 Step best = 0.04
RMSE = 4.19669887508 Step = 0.05 RMSE Best = 4.19669887508 Step best = 0.05
Final RMSE(Normalization) = 4.26217761602  Best Step size = 0.05

'''