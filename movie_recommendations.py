"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * 
 * Problems -1. Movie recommendation
 *           2. input movies can be incorrectly spelled so use Levenshtein
 *              distance for best match
 * ********************************************************************************************/

"""


from pyspark import SparkConf, SparkContext
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql import SQLContext, DataFrame
from pyspark.sql.functions import split, levenshtein
import sys, string
from os.path import join
from pyspark.mllib.recommendation import ALS

""" sbaronia - this function parses movies and ratings data
to required format"""
def parse_rating_movie(line, whatdata):
	
	if (whatdata == "movies.dat"):
		elements = line.strip().split("::")
		return int(elements[0]), elements[1].encode('ascii', 'ignore')

	elif (whatdata == "ratings.dat"):
		elements = line.strip().split("::")
		return (int(elements[0]), int(elements[1]), float(elements[2]))

	else:
		print("File not found. Provide correct file")
		sys.exit()

""" sbaronia - this fuction parses input file to required format"""
def parse_my_input(line):
	elements = [line[:line.index(' ')] ,line[line.index(' ')+1:]]
	return 0, elements[1].encode('ascii', 'ignore'), int(elements[0]), 0


def main():
	inputs = sys.argv[1]
	rating_file = sys.argv[2]
	output = sys.argv[3]

	conf = SparkConf().setAppName('movie recommendation')
	sc = SparkContext(conf=conf)
	assert sc.version >= '1.5.1'

	sqlContext = SQLContext(sc)
	
	""" sbaronia - getting files from directory and 
	reading from it and using parse_rating_movie and parse_my_input for parsing the
	content of the files to an rdd"""

	movies_path = join(inputs, "movies.dat")
	ratings_path = join(inputs, "ratings.dat")
	
	read_ratings = sc.textFile(ratings_path)
	read_movies  = sc.textFile(movies_path)
	read_mymovies = sc.textFile(rating_file)

	parse_ratings = read_ratings.map(lambda line : parse_rating_movie(line, "ratings.dat")).cache()
	parse_movies = read_movies.map(lambda line : parse_rating_movie(line, "movies.dat")).cache()
	parse_mymovies = read_mymovies.map(lambda line: parse_my_input(line)).cache()
	
	""" sbaronia - converting movie and rating data to dataframes """

	schema_movie = StructType([StructField('movie_id', IntegerType(), True),
								StructField('movie_name', StringType(), True)])

	movie_df = sqlContext.createDataFrame(parse_movies, schema=schema_movie).cache()


	schema_mymovie = StructType([StructField('ip_uid', IntegerType(), True),
								StructField('ip_mname', StringType(), True),
								StructField('ip_rating', IntegerType(), True),
								StructField('ldistance', IntegerType(), True)])

	mymovie_df = sqlContext.createDataFrame(parse_mymovies, schema=schema_mymovie).cache()

	""" sbaronia - combining user input movies with movies data
	then finding Levenshtein distance with every movie and then finding
	the one with minimum Levenshtein distance as our best match"""

	movie_plus_ip = movie_df.join(mymovie_df, None, 'inner').cache()
		
	movie_plus_ip_distance = movie_plus_ip.withColumn('ldistance', levenshtein('movie_name','ip_mname'))

	mymovie_distance = movie_plus_ip_distance \
							  .groupBy('ip_uid', 'ip_mname') \
							  .min('ldistance') \
							  .withColumnRenamed('min(ldistance)','ldistance') \
							  .cache()

	""" sbaronia - join the tables to get only those movies with minimum 
	Levenshtein distance and then from that table select columns 
	necessary. Then create a test data for all movies with new user 0"""
	refined_movies = movie_plus_ip_distance.join(mymovie_distance, ['ip_uid', 'ip_mname', 'ldistance'], 'inner').cache()
	
	input_rating = refined_movies.select('ip_uid', 'movie_id', 'ip_rating').cache()

	input_rating_rdd = input_rating.rdd.map(lambda row1: (row1.ip_uid, row1.movie_id, float(row1.ip_rating))).cache()
	
	input_with_train = sc.union([input_rating_rdd, parse_ratings]).cache()
	
	test_newuser = parse_movies.map(lambda line: (0, line[0])).cache()
	
	""" sbaronia - train on all data including new one and then 
	test on all movies for new user and sort them in descending 
	order of ratings"""
	model = ALS.train(input_with_train, 10, 10, 0.1)	
	predictions = model.predictAll(test_newuser) \
					   .map(lambda row1: (row1.rating, row1.product)) \
					   .sortByKey(ascending=False) \
					   .map(lambda row: (row[1], row[0])) \
					   .cache()

	final_rating = sqlContext.createDataFrame(predictions, ['movie_id', 'movie_rating']).cache()

	final_movie_rating = movie_df.join(final_rating, ['movie_id'], 'inner').sort("movie_rating", ascending=False).cache()

	final_movie_rating_rdd = final_movie_rating.rdd.map(lambda row: (str(row.movie_id) + ' :: ' + str(row.movie_name)) + ' :: ' + str(row.movie_rating)).coalesce(1).cache()
	final_movie_rating_rdd.saveAsTextFile(output)
			
	
if __name__ == "__main__":
	main()



"""
Intermediate output -
=====================

ratings table
+-------+--------+------+
|user_id|movie_id|rating|
+-------+--------+------+
|      1|   68646|    10|
|      1|  113277|    10|
|      2|  422720|     8|
|      2|  454876|     8|
|      2|  790636|     7|
|      2|  816711|     8|
|      2| 1091191|     7|
|      2| 1103275|     7|
|      2| 1322269|     7|

movie table

+--------+--------------------+
|movie_id|          movie_name|
+--------+--------------------+
|       8|Edison Kinetoscop...|
|      10|La sortie des usi...|
|      12|The Arrival of a ...|
|      91|Le manoir du diab...|
|     417|Le voyage dans la...|
|     439|The Great Train R...|
|     628|The Adventures of...|
|     833|The Country Docto...|
|    1223| Frankenstein (1910)|
|    1740|The Lonedale Oper...|
|    2101|    Cleopatra (1912)|
|    2130|    L'inferno (1911)|
|    2844|Fantmas -  l'ombr...|

input by user -
+-----+--------------------+---------+---------+
|ip_uid|            ip_mname|ip_rating|ldistance|
+-----+--------------------+---------+---------+
|    0|The Lord of the R...|       10|        0|
|    0|The Lord of the R...|        8|        0|
|    0|  orrest Gump (1994)|        5|        0|
|    0|Mad Max: Fury Roa...|        9|        0|
|    0|      Mad Max (1979)|        3|        0|
+-----+--------------------+---------+---------+


+--------+--------------------+-----+--------------------+---------+---------+
|movie_id|          movie_name|ip_uid|            ip_mname|ip_rating|ldistance|
+--------+--------------------+-----+--------------------+---------+---------+
|       8|Edison Kinetoscop...|    0|The Lord of the R...|       10|        0|
|       8|Edison Kinetoscop...|    0|The Lord of the R...|        8|        0|
|      10|La sortie des usi...|    0|The Lord of the R...|       10|        0|
|      10|La sortie des usi...|    0|The Lord of the R...|        8|        0|
|      12|The Arrival of a ...|    0|The Lord of the R...|       10|        0|
|      12|The Arrival of a ...|    0|The Lord of the R...|        8|        0|
|      91|Le manoir du diab...|    0|The Lord of the R...|       10|        0|
|      91|Le manoir du diab...|    0|The Lord of the R...|        8|        0|
|     417|Le voyage dans la...|    0|The Lord of the R...|       10|        0|
|     417|Le voyage dans la...|    0|The Lord of the R...|        8|        0|
|     439|The Great Train R...|    0|The Lord of the R...|       10|        0|
|     439|The Great Train R...|    0|The Lord of the R...|        8|        0|
|     628|The Adventures of...|    0|The Lord of the R...|       10|        0|
|     628|The Adventures of...|    0|The Lord of the R...|        8|        0|
|     833|The Country Docto...|    0|The Lord of the R...|       10|        0|
|     833|The Country Docto...|    0|The Lord of the R...|        8|        0|
|    1223| Frankenstein (1910)|    0|The Lord of the R...|       10|        0|


we are find going to join movie table with input table making a new column, input rating and another column to find levenshtein distance

make something like

movie_id | movie_name | ip_movie_name | ip_rating | levenshtein

+--------+--------------------+-----+--------------------+---------+---------+
|movie_id|          movie_name|ip_uid|            ip_mname|ip_rating|ldistance|
+--------+--------------------+-----+--------------------+---------+---------+
|       8|Edison Kinetoscop...|    0|The Lord of the R...|       10|       38|
|       8|Edison Kinetoscop...|    0|The Lord of the R...|        8|       42|
|      10|La sortie des usi...|    0|The Lord of the R...|       10|       35|
|      10|La sortie des usi...|    0|The Lord of the R...|        8|       41|
|      12|The Arrival of a ...|    0|The Lord of the R...|       10|       34|
|      12|The Arrival of a ...|    0|The Lord of the R...|        8|       40|
|      91|Le manoir du diab...|    0|The Lord of the R...|       10|       38|
|      91|Le manoir du diab...|    0|The Lord of the R...|        8|       45|
|     417|Le voyage dans la...|    0|The Lord of the R...|       10|       33|
|     417|Le voyage dans la...|    0|The Lord of the R...|        8|       41|
|     439|The Great Train R...|    0|The Lord of the R...|       10|       33|
|     439|The Great Train R...|    0|The Lord of the R...|        8|       41|
|     628|The Adventures of...|    0|The Lord of the R...|       10|       35|
|     628|The Adventures of...|    0|The Lord of the R...|        8|       38|
|     833|The Country Docto...|    0|The Lord of the R...|       10|       35|

+-----+--------------------+---------+
|ip_uid|            ip_mname|ldistance|
+-----+--------------------+---------+
|    0|The Lord of the R...|        7|
|    0|  orrest Gump (1994)|        1|
|    0|      Mad Max (1979)|        0|
|    0|The Lord of the R...|        0|
|    0|Mad Max: Fury Roa...|        1|
+-----+--------------------+---------+

+--------+--------------------+-----+--------------------+---------+---------+
|movie_id|          movie_name|ip_uid|            ip_mname|ip_rating|ldistance|
+--------+--------------------+-----+--------------------+---------+---------+
|  120737|The Lord of the R...|    0|The Lord of the R...|        8|        0|
|  167260|The Lord of the R...|    0|The Lord of the R...|       10|        7|
|   79501|      Mad Max (1979)|    0|      Mad Max (1979)|        3|        0|
|  109830| Forrest Gump (1994)|    0|  orrest Gump (1994)|        5|        1|
| 1392190|Mad Max: Fury Roa...|    0|Mad Max: Fury Roa...|        9|        1|
+--------+--------------------+-----+--------------------+---------+---------+


+-----+--------+---------+
|ip_uid|movie_id|ip_rating|
+-----+--------+---------+
|    0|  120737|        8|
|    0|  167260|       10|
|    0|   79501|        3|
|    0|  109830|        5|
|    0| 1392190|        9|
+-----+--------+---------+



[Rating(user=0, product=167260, rating=8.477759422937254), 
Rating(user=0,  product=79501, rating=3.008795821438842), 
Rating(user=0,  product=120737, rating=8.274334510878374), 
Rating(user=0,  product=109830, rating=6.363582831983525), 
Rating(user=0,  product=1392190, rating=8.404572868552801)]

+--------+------------------+
|movie_id|      movie_rating|
+--------+------------------+
| 3700594| 16.89631733785954|
| 1648201|16.737405145147076|
| 2375037|16.096384196038947|
| 2750632|15.904646618558123|
| 4107858| 15.67869611544623|
| 3552336|15.626605785550957|
| 2788512|15.124241127185956|
|  145464|14.814411533745382|


+--------+--------------------+------------------+
|movie_id|          movie_name|      movie_rating|
+--------+--------------------+------------------+
| 3045760|          Kil (2013)|19.109809511786665|
| 2993250|Donner Party: The...|17.330753459507502|
| 1130965|       Atletu (2009)| 16.29834307036766|
| 3243554| Four Corners (2013)|16.267434137711035|
| 5027202|Colombia magia sa...|15.668289388412992|
|  166813|Spirit: Stallion ...|15.445945061440383|
|  447431|Living Luminaries...|15.281043528490143|
| 3008180|  Coming Home (2013)| 15.08071022781469|
|  106471|Boxing Helena (1993)|14.834807859651846|
| 2375037|  Full Circle (2013)|14.825187842228154|
|  221542|Sentimientos: Mir...|14.740850406038234|
| 3264046|         Silo (2013)|14.729494857785788|
| 3429304| Attic Entity (2014)|14.696769079317328|
| 1975158|  Finding Joy (2013)| 14.53214329771708|
| 1439527|       Burden (2009)| 14.53214329771708|

"""

