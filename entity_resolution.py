"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 *
 * Problems - 1. Convet records to tokenset, remove stop words
 *            2. Filter non matching pairs with atleast one matching requirement
 *            3. Calculate Jaccard similarity of remaining matched pairs
 *            4. evaluate Precision, recall and fmeasure
 * ********************************************************************************************/ 
"""


# entity_resolution.py
import re
import operator
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.sql.functions import concat_ws

conf = SparkConf().setAppName('Entity Resolution')
sc = SparkContext(conf=conf)
assert sc.version >= '1.5.1'
sqlCt = SQLContext(sc)

'''sbaronia - global list to hold stop words'''
stop_words = []

'''sbaronia - this function cleans the joinkey from stop words'''
def lineTokenizer(line):
    line = line.lower()
    clean_words = []
    string_set = re.split(r'\W+', line)

    for word in set(string_set):
        if (word in stop_words or word == '') :
            string_set.remove(word)

    return string_set

'''sbaronia - for every id joinkey pair take every token
and make its tuple with id and maintain a list'''
def token_id_pair(line):
    list_x = []
    for t in line.joinKey:
        list_x.append((line.id,t))
    return list_x

'''sbaronia - calculation of jaccard value using two keys
with edge cases addressed'''
def jaccard_cal(key1,key2):
    len_1 = len(set(key1))
    len_2 = len(set(key2))
    len_12 = len(set(key1).intersection(set(key2)))
    if (len_1 > 0 or len_2 > 0):
        if len_12 == 0:
            jaccard = 0
        else:
            jaccard = float(len_12)/(len_1 + len_2 - len_12)
    else:
        jaccard = 0

    return jaccard

class EntityResolution:
    def __init__(self, dataFile1, dataFile2, stopWordsFile):
        self.f = open(stopWordsFile, "r")
        self.stopWords = set(self.f.read().split("\n"))
        self.stopWordsBC = sc.broadcast(self.stopWords).value
        self.df1 = sqlCt.read.parquet(dataFile1).cache()
        self.df2 = sqlCt.read.parquet(dataFile2).cache()
        '''sbaronia - save stop words in our global list'''
        global stop_words 
        stop_words = self.stopWordsBC

    '''sbaronia - make joinkeys from passed columns, clean them of stop words and return
    a new dataframe with joinkey as new column'''
    def preprocessDF(self, df, cols): 
        tokenize_udf = udf(lambda line: lineTokenizer(line),ArrayType(StringType(), False))
        df_joinkey = df.withColumn("joinKey", tokenize_udf(concat_ws(' ', cols[0], cols[1]).alias('joinKey'))).cache() #can we remove hardcoding of the cols!!!!!
        return df_joinkey

    '''sbaronia - join amazon and google tables based on the condition if
    an entry in amazon table shares atleast one token with an entry in google table.
    This is done by making token and id dataframe for both cases and join token column
    to original dataframe and then do final join on both dataframes and return a dataframe
    with id and joinkey for both tables with shared tokens'''
    def filtering(self, df1, df2):
        rdd1 = df1.select("id", "joinKey").map(token_id_pair).flatMap(lambda line: line).cache()
        rdd2 = df2.select("id", "joinKey").map(token_id_pair).flatMap(lambda line: line).cache()
        df1_temp = sqlCt.createDataFrame(rdd1, ["id","token"])
        df2_temp = sqlCt.createDataFrame(rdd2, ["id","token"])

        df1 = df1.join(df1_temp, df1.id == df1_temp.id, 'inner') \
                 .drop(df1_temp.id) \
                 .withColumnRenamed('id','id1') \
                 .withColumnRenamed('joinKey', 'joinKey1') \
                 .select('id1','joinKey1','token') \
                 .cache()

        df2 = df2.join(df2_temp, df2.id == df2_temp.id, 'inner') \
                 .drop(df2_temp.id) \
                 .withColumnRenamed('id','id2') \
                 .withColumnRenamed('joinKey', 'joinKey2') \
                 .select('id2','joinKey2','token') \
                 .cache()

        candDF = df1.join(df2, df1.token == df2.token, 'inner') \
                   .drop('token') \
                   .dropDuplicates() \
                   .cache()

        return candDF

    '''sbaronia - find jaccard value using both joinkeys columns and add it as 
    a new column and filter results less than a threshold'''
    def verification(self, candDF, threshold):
        jaccard_udf = udf(jaccard_cal, FloatType())
        resultDF = candDF.withColumn("jaccard", jaccard_udf(candDF.joinKey1,candDF.joinKey2)).cache()
        resultDF = resultDF.filter(resultDF.jaccard >= threshold).cache() #keeping those equal to threshold

        return resultDF

    '''sbaronia - calculate prcesion, recall and fmesaure from obtained and 
    groundTruth results'''
    def evaluate(self, result, groundTruth):
        tru_match_res = len(set(result).intersection(set(groundTruth)))
        ident_matching = len(result)
        tru_match_ent = len(groundTruth)

        if (ident_matching == 0 or tru_match_ent == 0 or tru_match_res == 0):
            precision = 0
            recall = 0
            fmesaure = 0
        else :
            precision = float(tru_match_res)/ident_matching
            recall = float(tru_match_res)/tru_match_ent
            fmesaure = float(2*precision*recall)/(precision+recall)

        return (precision,recall,fmesaure)


    def jaccardJoin(self, cols1, cols2, threshold):
        newDF1 = self.preprocessDF(self.df1, cols1)
        newDF2 = self.preprocessDF(self.df2, cols2)
        print "Before filtering: %d pairs in total" %(self.df1.count()*self.df2.count()) 

        candDF = self.filtering(newDF1, newDF2)
        print "After Filtering: %d pairs left" %(candDF.count())

        resultDF = self.verification(candDF, threshold)
        print "After Verification: %d similar pairs" %(resultDF.count())

        return resultDF


    def __del__(self):
        self.f.close()


if __name__ == "__main__":
    er = EntityResolution("amazon-google-sample/Amazon_sample", "amazon-google-sample/Google_sample", "amazon-google-sample/stopwords.txt")
    #er = EntityResolution("amazon-google/Amazon", "amazon-google/Google", "amazon-google/stopwords.txt")
    amazonCols = ["title", "manufacturer"]
    googleCols = ["name", "manufacturer"]
    resultDF = er.jaccardJoin(amazonCols, googleCols, 0.5)

    result = resultDF.map(lambda row: (row.id1, row.id2)).collect()
    groundTruth = sqlCt.read.parquet("amazon-google-sample/Amazon_Google_perfectMapping_sample") \
                          .map(lambda row: (row.idAmazon, row.idGoogle)).collect()
    #groundTruth = sqlCt.read.parquet("amazon-google/Amazon_Google_perfectMapping") \
    #                      .map(lambda row: (row.idAmazon, row.idGoogle)).collect()
    print "(precision, recall, fmeasure) = ", er.evaluate(result, groundTruth)


'''
Result -

With sample
--------------

Before filtering: 256 pairs in total
After Filtering: 79 pairs left
After Verification: 6 similar pairs
(precision, recall, fmeasure) =  (1.0, 0.375, 0.5454545454545454)


Actual non-sample
--------------------

Before filtering: 4397038 pairs in total
After Filtering: 690126 pairs left
After Verification: 2031 similar pairs
(precision, recall, fmeasure) =  (0.36976858690300346, 0.5776923076923077, 0.4509156409486641)
'''