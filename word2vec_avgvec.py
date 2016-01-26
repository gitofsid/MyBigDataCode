"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 *
 * Problems - 1. Find average word2vec vector for a review in corpus
 *            2. Using overall rating for every review run LinearRegressionWithSGD on data
 * ********************************************************************************************/ 
"""

import sys,json,string,random
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, DataFrame
from pyspark.mllib.feature import Word2Vec,Word2VecModel
from pyspark.mllib.clustering import KMeans, KMeansModel

'''sbaronia - zip an rdd with index'''
def rdd_zip(rdd):
	return rdd.zipWithIndex().cache()

'''sbaronia - this function extracts year from reviewTime field
in json such as "06 18, 2013" = 2013 '''
def extract_year(review):
  line_year = int((review.reviewTime).split(', ')[1])
  return line_year

'''sbaronia - removes puntuation from line and '''
def clean_string_to_words(line):

	line_review = (line).lower()

	for st in string.punctuation:
		line_review = line_review.replace(st, ' ')

	words_list = line_review.lower().split(' ')	
	words_list = filter(None,words_list)

	return words_list

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

'''sbaronia - run LinearRegressionWithSGD on training and testing
and return root mean square error'''
def regression_and_error(train_x,test_x,step_x):
  model = LinearRegressionWithSGD.train(train_x, iterations=500, step=step_x)
  valpred = test_x.map(lambda test: (test.label, model.predict(test.features))).cache()
    
  rmse = math.sqrt(valpred.map(lambda (v, p): (v-p)**2).reduce(lambda x, y: x + y)/valpred.count())   
  return rmse

def main():
  input_model = sys.argv[1] #model to be read
  input_file = sys.argv[2] #review file to be read

  conf = SparkConf().setAppName('Word2Vec')
  sc = SparkContext(conf=conf)
  assert sc.version >= '1.5.1'

  sqlContext = SQLContext(sc)

  '''sbaronia - load word2vec model last saved'''
  word2vec_model = Word2VecModel.load(sc,input_model)

  '''sbaronia - get three fields from json and make data frame with index'''
  review = sqlContext.read.json(input_file).select('reviewText','overall','reviewTime').cache()
  review_df = review.filter(review.reviewText != "").cache()

  rating_rdd = rdd_zip(review_df.map(lambda line: float(line.overall)).cache()).cache()
  rating_df = sqlContext.createDataFrame(rating_rdd, ['rating', 'index']).cache()

  year_rdd = rdd_zip(review_df.map(extract_year).cache()).cache()
  year_df = sqlContext.createDataFrame(year_rdd, ['year', 'index']).cache()

  clean_words_rdd = review_df.map(lambda review: clean_string_to_words(review.reviewText)).cache()

  clean_list = clean_words_rdd.collect()

  '''sbaronia - make a list of all words in our model'''
  keys = sqlContext.read.parquet(input_model+"/data")
  keys_list = keys.rdd.map(lambda line: line.word).collect()

  '''sbaronia - using loaded model find vector for every word in review
  sum them and find average vector for a review'''
  avg_vec = []
  for i in range(len(clean_list)):
    sum_init = 0
    count = 0
    for word in clean_list[i]:
      if word in keys_list:
      	count = count + 1
      	vec = word2vec_model.transform(word)
      	sum_init = sum_init + vec
    if count > 0:
      avg_vec.append(sum_init/count)  

  '''sbaronia - create an rdd of this avg vector for all reviews'''
  avg_vec_rdd = rdd_zip(sc.parallelize(avg_vec).cache()).cache()
  avg_vec_df = sqlContext.createDataFrame(avg_vec_rdd, ['vector', 'index']).cache()

  '''sbaronia - make a dataframe with overall rating and avg vector'''
  year_rating_df = rating_df.join(year_df, rating_df.index == year_df.index, 'outer').drop(rating_df.index).cache()
  vecyearrate_df = avg_vec_df.join(year_rating_df, avg_vec_df.index == year_rating_df.index, 'inner') \
                             .drop(avg_vec_df.index).cache()

  '''sbaronia - extract training and testing rdd based on year'''
  train_rdd = vecyearrate_df.filter(vecyearrate_df.year < 2014) \
                          .select('rating','vector') \
                          .map(lambda line: (LabeledPoint(line.rating, line.vector))) \
                          .coalesce(1) \
                          .cache()
  
  test_rdd = vecyearrate_df.filter(vecyearrate_df.year == 2014) \
                         .select('rating','vector') \
                         .map(lambda line: (LabeledPoint(line.rating, line.vector))) \
                         .coalesce(1) \
                         .cache()

  '''sbaronia - find best step using validation and run regression to get final RMSE'''
  step_best = validation(train_rdd)

  RMSE = regression_and_error(train_rdd,test_rdd,step_best)

  print("Final RMSE = " + str(RMSE) + "  Best Step size = " + str(step_best))


if __name__ == "__main__":
	main()


'''

To run on local - 

spark-submit --master local[*] word2vec_q6.py output45/word2vec/ data/reviews_Pet_Supplies_p1.json

Final RMSE = 2.42343398732 Best Step size = 0.05

Output -


'''