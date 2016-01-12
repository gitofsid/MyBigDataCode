"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 *
 * Problems - 1. Use ML lib and methods for classification with pre defined params
 *            2. Using cross validation with sets of params to find best fit 
 *            3. Reclassify and test on testing data to get result with best found model
 * 
 * ********************************************************************************************/ 
"""

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator


conf = SparkConf().setAppName("MLPipeline")
sc = SparkContext(conf=conf)

# Read training data as a DataFrame
sqlCt = SQLContext(sc)
trainDF = sqlCt.read.parquet("20news_train.parquet")

# Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features", numFeatures=1000)
lr = LogisticRegression(maxIter=20, regParam=0.1)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

# Fit the pipeline to training data.
model = pipeline.fit(trainDF)

# Evaluate the model on testing data
testDF = sqlCt.read.parquet("20news_test.parquet")
prediction = model.transform(testDF)
evaluator = BinaryClassificationEvaluator()
print evaluator.evaluate(prediction)


'''sbaronia - setting up parameters using 
ParamGridBuilder with 3 different features and 9 diff regParam'''
param_Grid = (ParamGridBuilder()
			.addGrid(hashingTF.numFeatures, [1000, 5000, 10000])
			.addGrid(lr.regParam, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
			.build())

'''sbaronia - creating a new CrossValidator that will
use above parameters and use same evaluator with 2 folds 
cross validation'''
cross_val = (CrossValidator()
			.setEstimator(pipeline)
			.setEvaluator(evaluator)
			.setEstimatorParamMaps(param_Grid)
			.setNumFolds(2))

'''sbaronia - running the cross validation and use best params'''
cross_val_model = cross_val.fit(trainDF)

'''sbaronia - use above found model on test data'''
cross_val_prediction = cross_val_model.transform(testDF)

'''sbaronia - print the areaUnderROC with tuning'''
print evaluator.evaluate(cross_val_prediction)

