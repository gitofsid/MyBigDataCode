"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 *
 * Problems - 1. Read training and testng data
 *            2. From training data for every movie pair with comon raters find deviation
 *            3. For every user and movie pair in testing predict rating
 *            4. Find rmse of predictions using slopeone technique
 * ********************************************************************************************/ 
"""


import sys, string, math
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

'''sbaronia - global dictionaries'''
mid_countdev_dict = [] #(mid1,mid2) > (count,deviation)
rating_dict = [] #(uid,mid) > rating
movieid_dict = [] #uid > mid

'''sbaronia - calculate prediction for every test user movie pair'''
def test_predict(uid,mid):
    set_n = movieid_dict.get(uid) #we have i to traverse
    num = 0.0
    den = 0.0

    for i in set_n:
        count = mid_countdev_dict.get((mid,i),(0,0))[0]
        num = num + (mid_countdev_dict.get((mid,i),(0,0))[1] + rating_dict.get((uid,i),0))*count
        den = den + count

    if den != 0:
        return float(num)/den
    else:
        return 0.0

def main():

    conf = SparkConf().setAppName('movie recommendation slopeone')
    sc = SparkContext(conf=conf)
    assert sc.version >= '1.5.1'
    sqlContext = SQLContext(sc)

    '''sbaronia - global dictionaries declarations'''
    global rating_dict
    global mid_countdev_dict
    global movieid_dict
    
    train_data = sc.textFile("movierecommendation/MovieLens100K_train.txt")
    test_data = sc.textFile("movierecommendation/MovieLens100K_test.txt")

    '''sbaronia - read training and testing data'''
    train_rating = train_data.map(lambda line: line.split('\t')) \
                             .map(lambda line: (int(line[0]), int(line[1]), int(line[2]))) \
                             .cache()


    test_rating = test_data.map(lambda line: line.split('\t')) \
                             .map(lambda line: (int(line[0]), int(line[1]), int(line[2]))) \
                             .cache()
    '''sbaronia - duplicate training dataframe and self join them 
    on condition that their uid are same but not mid'''
    train_df_1 = sqlContext.createDataFrame(train_rating, ['uid', 'mid1', 'rating1']).cache()
    train_df_2 = sqlContext.createDataFrame(train_rating, ['uid', 'mid2', 'rating2']).cache()

    cond = [train_df_1.uid == train_df_2.uid, train_df_1.mid1 != train_df_2.mid2]
    self_join = train_df_1.join(train_df_2, cond, 'inner') \
                  .drop(train_df_2.uid) \
                  .select('mid1','mid2','rating1','rating2','uid') \
                  .cache()

    '''sbaronia - find difference in rating for each movie pair and add a column'''
    sjoin_diff = self_join.withColumn('rdifference', self_join.rating1 - self_join.rating2).cache()

    '''sbaronia - find mean of all movie pairs  rated by common raters to get deviation'''
    sjoin_dev = sjoin_diff.groupBy('mid1','mid2') \
                          .mean('rdifference') \
                          .withColumnRenamed('avg(rdifference)','deviation') \
                          .cache()

    '''sbaronia - find count of every movie pair rated'''
    sjoin_count = sjoin_diff.groupBy('mid1','mid2').count().cache()
    
    '''sbaronia - join df with deviation and count'''
    cond1 = [sjoin_count.mid1 == sjoin_dev.mid1, sjoin_count.mid2 == sjoin_dev.mid2]
    sjoin_c_d = sjoin_count.join(sjoin_dev, cond1, 'inner') \
                           .withColumnRenamed("count", "count_1") \
                           .drop(sjoin_dev.mid1) \
                           .drop(sjoin_dev.mid2) \
                           .dropDuplicates() \
                           .cache()

    '''sbaronia - create dictionaries from training data and new dataframe created 
    mid_countdev_dict - (mid1,mid2) : (count,deviation)
    rating_dict - (uid,mid) : rating
    movieid_dict - uid : mid'''
    mid_countdev_dict = sjoin_c_d.map(lambda line: ((line[1],line[2]),(line[0],line[3]))).collectAsMap()
    rating_dict = train_rating.map(lambda line: ((line[0],line[1]),line[2])).collectAsMap()
    movieid_dict = train_rating.map(lambda line: (line[0],line[1])).groupByKey().mapValues(list).collectAsMap()
    
    test_df = sqlContext.createDataFrame(test_rating, ['uid','mid','rating']).cache()

    '''sbaronia - for every mid and uid pair in test find prediction as a new column'''
    predict_udf = udf(test_predict, FloatType())
    predict_df = test_df.withColumn('prediction', predict_udf(test_df.uid,test_df.mid)).cache()

    '''sbaronia - find rmse'''
    MSE = predict_df.map(lambda r: (r[2] - r[3])**2).reduce(lambda x,y: x + y)
    RMSE = float(math.sqrt(MSE/predict_df.count()))

    print("RMSE = " + str(RMSE))

    
if __name__ == "__main__":
    main()



'''
RMSE = 0.945089793509
'''