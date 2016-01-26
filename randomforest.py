
"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 *
 * Problems - 1. Run regression on the data using RandomForest regression model
 * 
 * ********************************************************************************************/ 
"""

import sys,math
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest

'''sbaronia - returns labeledpoint for every row read'''
def to_labeledpoint(line):
    line_spl = line.split(' :: ')
    return LabeledPoint(line_spl[0], SparseVector.parse(line_spl[1]))

def main():
    input_train = sys.argv[1]
    input_test = sys.argv[2]

    conf = SparkConf().setAppName('Sentiment Analysis with Random Forest')
    sc = SparkContext(conf=conf)
    assert sc.version >= '1.5.1'

    train = sc.textFile(input_train).cache()
    test = sc.textFile(input_test).cache()

    '''sbaronia - get training and testing labeled points'''
    train_lp = train.map(to_labeledpoint).cache()
    test_lp = test.map(to_labeledpoint).cache()

    '''sbaronia - run RandomForest regression on our training data with
    default options except numTrees = 5'''
    rf_model = RandomForest.trainRegressor(train_lp,categoricalFeaturesInfo={},numTrees=5,featureSubsetStrategy="auto", impurity='variance', maxDepth=4, maxBins=32)
    
    '''sbaronia - run predictions on testing data and calculate RMSE value'''
    predictions = rf_model.predict(test_lp.map(lambda x: x.features))
    labelsAndPredictions = test_lp.map(lambda lp: lp.label).zip(predictions)
    rmse = math.sqrt(labelsAndPredictions.map(lambda (v, p): (v-p)**2).reduce(lambda x, y: x + y)/float(test_lp.count()))

    print("RMSE = " + str(rmse))


if __name__ == "__main__":
    main()


'''
To run on local -

spark-submit --master local[*] randomforest_q8.py output-p1/train-text/part-00000 output-p1/test-text/part-00000

output:

RMSE = 2.23785982345

'''