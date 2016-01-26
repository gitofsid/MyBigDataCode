
"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 *
 * Problems - 1. Find cluster index for every word in a review and make a vector 
 *               from how many times that cluster appeared in a review 
 *            2. Make a L1 normalized SparseVector from above 
 *            3. Run LinearRegressionWithSGD on the data
 * ********************************************************************************************/ 
"""

import sys, operator, string,json, math
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, DataFrame
from pyspark.mllib.feature import Word2Vec,Word2VecModel,Normalizer
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.clustering import KMeansModel, KMeans

'''sbaronia - this function extracts year from reviewTime field
in json such as "06 18, 2013" = 2013 '''
def extract_year(review):
    line_year = int((review.reviewTime).split(', ')[1])
    return line_year

'''sbaronia - zip an rdd with index'''
def rdd_zip(rdd):
    return rdd.zipWithIndex().cache()

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
    k_input_model = sys.argv[1] #read kmean model from this location
    w_input_model = sys.argv[2] #read word2vec model from this location
    input_file = sys.argv[3] #read input file

    conf = SparkConf().setAppName('Clustering')
    sc = SparkContext(conf=conf)
    assert sc.version >= '1.5.1'

    sqlContext = SQLContext(sc)

    '''sbaronia - load both kmean and Word2Vec model'''
    kmean_model = KMeansModel.load(sc,k_input_model)
    word2vec_model = Word2VecModel.load(sc,w_input_model)

    '''sbaronia - select fields from json and make data frame zipped with index'''
    review = sqlContext.read.json(input_file).select('reviewText','overall','reviewTime').cache()
    review_df = review.filter(review.reviewText != "").cache()

    rating_rdd = rdd_zip(review_df.map(lambda line: float(line.overall)).cache()).cache()
    rating_df = sqlContext.createDataFrame(rating_rdd, ['rating', 'index']).cache()

    year_rdd = rdd_zip(review_df.map(extract_year).cache()).cache()
    year_df = sqlContext.createDataFrame(year_rdd, ['year', 'index']).cache()

    clean_words_rdd = review_df.map(lambda review: clean_string_to_words(review.reviewText)).cache()
       
    clean_list = clean_words_rdd.collect()

    '''sbaronia - make a list of all words in our model'''
    keys = sqlContext.read.parquet(w_input_model+"/data")
    keys_list = keys.rdd.map(lambda line: line.word).collect()

    '''sbaronia - here we create one vector per review, where vector
    contains the number of times a cluster is assinged to a word in
    a review. We make a SparseVector compatible format'''
    features = []

    for i in range(len(clean_list)):
        histogram = [0] * 2000
        for word in clean_list[i]:
            if word in keys_list:
                vec = word2vec_model.transform(word)
                clust = kmean_model.predict(vec)
                if histogram[clust] > 0:
                    histogram[clust] = histogram[clust] + 1
                else:
                    histogram[clust] = 1
        features.append((2000,range(2000),histogram))

    '''sbaronia - create a normalized SparseVector rdd'''
    nor = Normalizer(1)
    features_rdd = rdd_zip(sc.parallelize(features) \
                             .map(lambda line: nor.transform(SparseVector.parse(line))) \
                             .cache()).cache()

    '''sbaronia - make a dataframe with rating, year and vector per review'''
    features_df = sqlContext.createDataFrame(features_rdd, ['feature', 'index']).cache()

    year_rating_df = rating_df.join(year_df, rating_df.index == year_df.index, 'outer').drop(rating_df.index).cache()
    featyearrate_df = features_df.join(year_rating_df, features_df.index == year_rating_df.index, 'inner') \
                                 .drop(features_df.index).cache()
    
    '''sbaronia - create training and testing data based on year'''
    train_rdd = featyearrate_df.filter(featyearrate_df.year < 2014) \
                            .select('rating','feature') \
                            .map(lambda line: (LabeledPoint(line.rating, line.feature))) \
                            .coalesce(1) \
                            .cache()
    
    test_rdd = featyearrate_df.filter(featyearrate_df.year == 2014) \
                           .select('rating','feature') \
                           .map(lambda line: (LabeledPoint(line.rating, line.feature))) \
                           .coalesce(1) \
                           .cache()

    '''sbaronia - find best step using validation and run LinearRegressionWithSGD 
    with that step and report final RMSE'''
    step_best_norm = validation(train_rdd)

    RMSE_norm = regression_and_error(train_rdd,test_rdd,step_best_norm)

    print("Final RMSE(Normalization) = " + str(RMSE_norm) + "  Best Step size = " + str(step_best_norm))


if __name__ == "__main__":
    main()


'''
To run on local -

spark-submit --master local[*] word2vec_q6.py output45/kmean/ output45/word2vec/ data/reviews_Pet_Supplies_p1.json


Output -
Final RMSE(Normalization) = 4.23765412747  Best Step size = 0.05

'''