"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 *
 * Problems - 1. Model development with KMeans
 *            2. categorical to numerical features
 *            3. Adding anomaly score of each data point
 * ********************************************************************************************/ 
"""

# anomaly_detection.py
from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.sql import SQLContext, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.mllib.linalg import Vectors, DenseVector, VectorUDT
import operator

conf = SparkConf().setAppName('Anomaly Detection')
sc = SparkContext(conf=conf)
assert sc.version >= '1.5.1'

sqlCt = SQLContext(sc)

'''sbaronia - we replace categorical feature by 
numerical feature using hot encoding and return a list'''
def replaceCat2Num(raw,feat0,feat1):
    raw[1] = str(feat1.strip('[]'))
    raw[0] = str(feat0.strip('[]'))
    return raw

class AnomalyDetection():
    '''sbaronia - returns hot encoding for a column depending on number of
    distinct values that column has'''
    def oneHotEncoding(self, df, input_col):
        stringInd = StringIndexer(inputCol=input_col, outputCol="indexed")
        model = stringInd.fit(df)
        td = model.transform(df)
        encoder = OneHotEncoder(inputCol="indexed", outputCol="features", dropLast=False)
        final_encoding = encoder.transform(td).select(df.id, 'features').cache()
        
        conv_udf = udf(lambda line: Vectors.dense(line).tolist())
        final_encoding = final_encoding.select(df.id,conv_udf(final_encoding.features).alias("num_"+input_col)).cache()

        return final_encoding

    def readData(self, filename):
        self.rawDF = sqlCt.read.parquet(filename).cache()

    '''sbaronia - catergorial to numerical feature conversion'''
    def cat2Num(self, df, indices):
        '''sbaronia - extract the categorical data and make df out of it
        so oneHotEncoding can be run on them'''
        protocol_ind0 = df.select(df.id,df.rawFeatures[indices[0]].alias("features0")).cache()
        protocol_ind1 = df.select(df.id,df.rawFeatures[indices[1]].alias("features1")).cache()

        ind0_enc = self.oneHotEncoding(protocol_ind0,"features0").cache()
        ind1_enc = self.oneHotEncoding(protocol_ind1,"features1").cache()
        
        '''sbaronia - add those hot encoded features columns to original df'''
        int_join_1 = df.join(ind0_enc, ind0_enc.id == df.id, 'inner').drop(ind0_enc.id).cache()
        int_join_2 = int_join_1.join(ind1_enc, int_join_1.id == ind1_enc.id, 'inner').drop(int_join_1.id).cache()

        '''sbaronia - now create a new column features which has 
        converted vector form and drop rest columns'''
        comb_udf = udf(replaceCat2Num,StringType())
        int_join_2 = int_join_2.select(int_join_2.id,int_join_2.rawFeatures, \
                                       comb_udf(int_join_2.rawFeatures, \
                                       int_join_2.num_features0, \
                                       int_join_2.num_features1).alias("features")).cache()
        
        '''sbaronia - convert list of numerical features to DenseVector
        so they can be used in KMeans'''
        dense_udf = udf(lambda line: DenseVector.parse(line), VectorUDT())
        feat = int_join_2.select(int_join_2.id,int_join_2.rawFeatures,dense_udf(int_join_2.features).alias("features")).cache()
      
        return feat

    '''sbaronia - finding anomaly score'''
    def addScore(self, df):
        to_integer_udf = udf(lambda x: int(x), IntegerType())
        df = df.select(df.id,df.rawFeatures,df.features,to_integer_udf(df.prediction).alias('prediction')).cache()

        '''sbaronia - finding frequency of each cluster in a df'''
        clus_count =  df.groupBy('prediction').count().cache()
        clus_count_list = clus_count.collect()

        '''sbaronia - finding frequencies maximum and minimum frequent clusters'''
        max_cl = clus_count.groupBy().max('count').collect()
        min_cl = clus_count.groupBy().min('count').collect()

        N_max = max_cl[0][0]
        N_min = min_cl[0][0]

        '''sbaronia - calculate score for every row of df'''
        score_udf = udf(lambda line: float(N_max - clus_count_list[line][1])/(N_max - N_min))
        with_score = df.withColumn("score", score_udf(df.prediction)).cache()

        return with_score
     

    def readToyData(self):
        data = [(0, ["http", "udt", 0.4]), \
                (1, ["http", "udf", 0.5]), \
                (2, ["http", "tcp", 0.5]), \
                (3, ["ftp", "icmp", 0.1]), \
                (4, ["http", "tcp", 0.4])]
        schema = ["id", "rawFeatures"]
        self.rawDF = sqlCt.createDataFrame(data, schema)

    def detect(self, k, t):
        #Encoding categorical features using one-hot.
        df1 = self.cat2Num(self.rawDF, [0, 1]).cache()
        df1.show()

        #Clustering points using KMeans
        features = df1.select("features").rdd.map(lambda row: row[0]).cache()
        model = KMeans.train(features, k, maxIterations=40, runs=10, initializationMode="random", seed=20)

        #Adding the prediction column to df1
        modelBC = sc.broadcast(model)
        predictUDF = udf(lambda x: modelBC.value.predict(x), StringType())
        df2 = df1.withColumn("prediction", predictUDF(df1.features)).cache()
        df2.show()

        #Adding the score column to df2; The higher the score, the more likely it is an anomaly 
        df3 = self.addScore(df2).cache()
        df3.show()    

        return df3.where(df3.score > t)


if __name__ == "__main__":
    ad = AnomalyDetection()
    ad.readData('logs-features-sample')
    #ad.readData('logs-features')
    #ad.readToyData()
    anomalies = ad.detect(8, 0.97)
    #anomalies = ad.detect(3, 0.9)
    print anomalies.count()
    anomalies.show()



"""
On small test - 
+---+----------------+
| id|     rawFeatures|
+---+----------------+
|  0|[http, udt, 0.4]|
|  1|[http, udf, 0.5]|
|  2|[http, tcp, 0.5]|
|  3|[ftp, icmp, 0.1]|
|  4|[http, tcp, 0.4]|
+---+----------------+

+---+-------------+
| id|num_features0|
+---+-------------+
|  0|   [1.0, 0.0]|
|  1|   [1.0, 0.0]|
|  2|   [1.0, 0.0]|
|  3|   [0.0, 1.0]|
|  4|   [1.0, 0.0]|
+---+-------------+

+---+--------------------+
| id|       num_features1|
+---+--------------------+
|  0|[0.0, 0.0, 1.0, 0.0]|
|  1|[0.0, 0.0, 0.0, 1.0]|
|  2|[1.0, 0.0, 0.0, 0.0]|
|  3|[0.0, 1.0, 0.0, 0.0]|
|  4|[1.0, 0.0, 0.0, 0.0]|
+---+--------------------+

+---+----------------+--------------------+
| id|     rawFeatures|            features|
+---+----------------+--------------------+
|  0|[http, udt, 0.4]|[1.0,0.0,0.0,0.0,...|
|  1|[http, udf, 0.5]|[1.0,0.0,0.0,0.0,...|
|  2|[http, tcp, 0.5]|[1.0,0.0,1.0,0.0,...|
|  3|[ftp, icmp, 0.1]|[0.0,1.0,0.0,1.0,...|
|  4|[http, tcp, 0.4]|[1.0,0.0,1.0,0.0,...|
+---+----------------+--------------------+

+---+----------------+--------------------+----------+
| id|     rawFeatures|            features|prediction|
+---+----------------+--------------------+----------+
|  0|[http, udt, 0.4]|[1.0,0.0,0.0,0.0,...|         0|
|  1|[http, udf, 0.5]|[1.0,0.0,0.0,0.0,...|         5|
|  2|[http, tcp, 0.5]|[1.0,0.0,1.0,0.0,...|         4|
|  3|[ftp, icmp, 0.1]|[0.0,1.0,0.0,1.0,...|         1|
|  4|[http, tcp, 0.4]|[1.0,0.0,1.0,0.0,...|         3|
+---+----------------+--------------------+----------+


with logs-features-sample

494
+-----+--------------------+--------------------+----------+------------------+
|   id|         rawFeatures|            features|prediction|             score|
+-----+--------------------+--------------------+----------+------------------+
|23231|[tcp, S0, -0.1585...|[1.0,0.0,0.0,0.0,...|         3|0.9887714838060433|
|67232|[tcp, S0, -0.1585...|[1.0,0.0,0.0,0.0,...|         3|0.9887714838060433|
|75432|[tcp, S0, -0.1585...|[1.0,0.0,0.0,0.0,...|         3|0.9887714838060433|
| 3033|[tcp, SF, 10.8899...|[1.0,0.0,0.0,1.0,...|         2|               1.0|
|58234|[tcp, S0, -0.1585...|[1.0,0.0,0.0,0.0,...|         3|0.9887714838060433|
|42035|[tcp, S0, -0.1585...|[1.0,0.0,0.0,0.0,...|         3|0.9887714838060433|
|50435|[tcp, S0, -0.1585...|[1.0,0.0,0.0,0.0,...|         3|0.9887714838060433|
|15236|[tcp, S0, -0.1585...|[1.0,0.0,0.0,0.0,...|         3|0.9887714838060433|
|31236|[tcp, S0, -0.1585...|[1.0,0.0,0.0,0.0,...|         3|0.9887714838060433|
|38436|[tcp, S0, -0.1585...|[1.0,0.0,0.0,0.0,...|         3|0.9887714838060433|
|58436|[tcp, S0, -0.1585...|[1.0,0.0,0.0,0.0,...|         3|0.9887714838060433|
|67636|[tcp, S0, -0.1585...|[1.0,0.0,0.0,0.0,...|         3|0.9887714838060433|
|69437|[tcp, S0, -0.1585...|[1.0,0.0,0.0,0.0,...|         3|0.9887714838060433|
|95037|[tcp, S0, -0.1585...|[1.0,0.0,0.0,0.0,...|         3|0.9887714838060433|
|14439|[tcp, S0, -0.1585...|[1.0,0.0,0.0,0.0,...|         3|0.9887714838060433|
|33439|[tcp, S0, -0.1585...|[1.0,0.0,0.0,0.0,...|         3|0.9887714838060433|
|47439|[tcp, S0, -0.1585...|[1.0,0.0,0.0,0.0,...|         3|0.9887714838060433|
|68439|[tcp, S0, -0.1585...|[1.0,0.0,0.0,0.0,...|         3|0.9887714838060433|
|89839|[tcp, S0, -0.1585...|[1.0,0.0,0.0,0.0,...|         3|0.9887714838060433|
|96639|[tcp, S0, -0.1585...|[1.0,0.0,0.0,0.0,...|         3|0.9887714838060433|
+-----+--------------------+--------------------+----------+------------------+

with logs-features

4815
+------+--------------------+--------------------+----------+------------------+
|    id|         rawFeatures|            features|prediction|             score|
+------+--------------------+--------------------+----------+------------------+
|  7431|[tcp, S0, -0.1594...|[1.0,0.0,0.0,0.0,...|         4|0.9888700080464151|
|104031|[tcp, S0, -0.1594...|[1.0,0.0,0.0,0.0,...|         4|0.9888700080464151|
|131231|[tcp, S0, -0.1594...|[1.0,0.0,0.0,0.0,...|         4|0.9888700080464151|
|132031|[tcp, S0, -0.1594...|[1.0,0.0,0.0,0.0,...|         4|0.9888700080464151|
|186431|[tcp, S0, -0.1594...|[1.0,0.0,0.0,0.0,...|         4|0.9888700080464151|
|212831|[tcp, S0, -0.1594...|[1.0,0.0,0.0,0.0,...|         4|0.9888700080464151|
|252231|[tcp, S0, -0.1594...|[1.0,0.0,0.0,0.0,...|         4|0.9888700080464151|
|259231|[tcp, S0, -0.1594...|[1.0,0.0,0.0,0.0,...|         4|0.9888700080464151|
|299431|[tcp, S0, -0.1594...|[1.0,0.0,0.0,0.0,...|         4|0.9888700080464151|
|306231|[tcp, S0, -0.1594...|[1.0,0.0,0.0,0.0,...|         4|0.9888700080464151|
|336631|[tcp, S1, -0.1594...|[1.0,0.0,0.0,0.0,...|         4|0.9888700080464151|
|526031|[tcp, S0, -0.1594...|[1.0,0.0,0.0,0.0,...|         4|0.9888700080464151|
|562831|[tcp, S0, -0.1594...|[1.0,0.0,0.0,0.0,...|         4|0.9888700080464151|
|634631|[tcp, S0, -0.1594...|[1.0,0.0,0.0,0.0,...|         4|0.9888700080464151|
|689631|[tcp, S0, -0.1594...|[1.0,0.0,0.0,0.0,...|         4|0.9888700080464151|
|697231|[tcp, SF, -0.1594...|[1.0,0.0,0.0,1.0,...|         0|               1.0|
|750431|[tcp, S0, -0.1594...|[1.0,0.0,0.0,0.0,...|         4|0.9888700080464151|
|753231|[tcp, S0, -0.1594...|[1.0,0.0,0.0,0.0,...|         4|0.9888700080464151|
|769231|[tcp, S0, -0.1594...|[1.0,0.0,0.0,0.0,...|         4|0.9888700080464151|
|777231|[tcp, S0, -0.1594...|[1.0,0.0,0.0,0.0,...|         4|0.9888700080464151|
+------+--------------------+--------------------+----------+------------------+
"""