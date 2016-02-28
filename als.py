"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 *
 * Problems - 1. Read training and test data 
 *            2. Train a model using ALS and find RMSE using testing data
 *            3. For different value of ranks and Regularization param find RMSE 
 *            4. plot graphs for ranks vs RMSE for diff Regularization param
 * ********************************************************************************************/ 
"""


import sys, string, math
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.recommendation import ALS, Rating
import matplotlib.pyplot as plt

'''sbaronia - draw graphs here for ranks vs rmse for diff
Regularization param'''
def draw_graphs(rank,rmse,lambda_value):
    f,ax = plt.subplots()
    ax.plot(rank,rmse, marker='o', linestyle='--', color='b', label='rmse',linewidth=2)
    plt.xlabel('Rank',fontweight='bold')
    plt.ylabel('RMSE',fontweight='bold')
    plt.legend(loc='best')
    title = "Regularization Parameter: " +  str(lambda_value)
    plt.title(title,fontweight='bold')
    plt.show()
    return

def main():

    conf = SparkConf().setAppName('movie recommendation als')
    sc = SparkContext(conf=conf)
    assert sc.version >= '1.5.1'
    sqlContext = SQLContext(sc)
    
    train_data = sc.textFile("movierecommendation/MovieLens100K_train.txt")
    test_data = sc.textFile("movierecommendation/MovieLens100K_test.txt")

    '''sbaronia - get training and testing data in Rating format'''
    train_rating = train_data.map(lambda line: line.split('\t')) \
                             .map(lambda line: Rating(int(line[0]), int(line[1]), int(line[2]))) \
                             .cache()


    test_rating = test_data.map(lambda line: line.split('\t')) \
                            .map(lambda line: Rating(int(line[0]), int(line[1]), int(line[2]))) \
                            .cache()

    '''sbaronia - get user id and movie id for test set'''
    train_df = sqlContext.createDataFrame(train_rating).cache()
    test_set = test_data.map(lambda line: line.split('\t')) \
                        .map(lambda line: (int(line[0]), int(line[1]))) \
                        .cache()

    
    """ sbaronia - train on all training data and then 
    test on test set"""
    model = ALS.train(train_df,rank=10, lambda_=0.1)    
    predictions = model.predictAll(test_set) \
                       .map(lambda row: ((row[0], row[1]), row[2])) \
                       .cache()

    '''sbaronia - with predictions found, compute RMSE'''
    preds_rate = test_rating.map(lambda line: ((line[0], line[1]), line[2])).join(predictions).cache()
    MSE = preds_rate.map(lambda r: (r[1][0] - r[1][1])**2).reduce(lambda x,y: x + y)
    RMSE = float(math.sqrt(MSE/preds_rate.count()))

    print ("RMSE in original run = " + str(RMSE) + " Rank = 10" + " Reg Param = 0.1")

    ranks = [2, 4, 8, 16, 32, 64, 128, 256]
    lambdas = [0.01, 0.1]
    
    '''sbaronia - for 8 different values of rank and 2 values of
    Regularization param retrain ALS model and retest to get RMSE for 
    diff reg param and draw graphs'''
    for l in lambdas:
        rmse_list = []
        for rank in ranks:
            model = ALS.train(train_df,rank=rank, lambda_=l)    
            predictions = model.predictAll(test_set) \
                               .map(lambda row: ((row[0], row[1]), row[2])) \
                               .cache()

            preds_rate = test_rating.map(lambda line: ((line[0], line[1]), line[2])).join(predictions).cache()
            MSE = preds_rate.map(lambda r: (r[1][0] - r[1][1])**2).reduce(lambda x,y: x + y)
            RMSE = float(math.sqrt(MSE/preds_rate.count()))
            rmse_list.append(RMSE)
            print ("RMSE = " + str(RMSE) + " Rank = " + str(rank) + " Reg Param = " + str(l))
        draw_graphs(ranks,rmse_list,l)
        print("\n")
    
if __name__ == "__main__":
    main()


'''
RMSE in original run = 0.934973948726 Rank = 10 Reg Param = 0.1

RMSE = 0.947610437783 Rank = 2 Reg Param = 0.01
RMSE = 0.979642279386 Rank = 4 Reg Param = 0.01
RMSE = 1.0485996957 Rank = 8 Reg Param = 0.01
RMSE = 1.1420318568 Rank = 16 Reg Param = 0.01
RMSE = 1.21106587009 Rank = 32 Reg Param = 0.01
RMSE = 1.20147461851 Rank = 64 Reg Param = 0.01
RMSE = 1.16805584531 Rank = 128 Reg Param = 0.01
RMSE = 1.18332005792 Rank = 256 Reg Param = 0.01


RMSE = 0.937568266827 Rank = 2 Reg Param = 0.1
RMSE = 0.937912346465 Rank = 4 Reg Param = 0.1
RMSE = 0.934016848346 Rank = 8 Reg Param = 0.1
RMSE = 0.930950811948 Rank = 16 Reg Param = 0.1
RMSE = 0.931658531487 Rank = 32 Reg Param = 0.1
RMSE = 0.926402123966 Rank = 64 Reg Param = 0.1
RMSE = 0.924698220373 Rank = 128 Reg Param = 0.1
RMSE = 0.922818094379 Rank = 256 Reg Param = 0.1

'''