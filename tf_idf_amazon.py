"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 *
 * Problems - 1. TF-IDF for each review in the reviews file
 * 
 * ********************************************************************************************/ 
"""

import sys,nltk,json,string
from pyspark import SparkConf, SparkContext
from nltk.corpus import stopwords
from pyspark.sql import SQLContext, DataFrame
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.mllib.feature import HashingTF, IDF

''' sbaronia - this function returns a set of stop words in english'''
def stop_words_func(ntlk_path):
	nltk.data.path.append(ntlk_path)
	stop_words = set(stopwords.words("english"))
	return stop_words

'''sbaronia - this function extracts year from reviewTime field
in json such as "06 18, 2013" = 2013 '''
def extract_year(review):
	line_year = int((review.reviewTime).split(', ')[1])
	return line_year
	
'''sbaronia - this function calculates the tf-idf of rdd of words
in a review and returns it'''
def tf_idf_cal(words_rdd):
	hashingTF = HashingTF()
	tf = hashingTF.transform(words_rdd)

	idf = IDF().fit(tf)
	
	tfidf = idf.transform(tf).cache()

	tfidf_str = tfidf.map(lambda line: str(line)).cache()

	return tfidf_str

'''sbaronia - this function zips an rdd with an index'''
def rdd_zip(rdd):
	return rdd.zipWithIndex().cache()

'''sbaronia - removes puntuation from line and 
removes stop words from review'''
def clean_string_to_words(line,stop_words):
	clean_words = []

	line_review = line.lower()

	for st in string.punctuation:
		line_review = line_review.replace(st, ' ')

	words_list = line_review.split(' ')
	
	for word in words_list:
		if word not in stop_words:
			clean_words.append(word)

	clean_words = filter(None,clean_words)

	'''sbaronia - some reviews are made only of punctuation
	and stop words like A+ so put null instead and it 
	will be filtered later'''
	if len(clean_words) == 0:
		clean_words.append('null')

	return clean_words

def main():
	inputs = sys.argv[1]
	output = sys.argv[2] 
	ntlk_path = sys.argv[3]

	conf = SparkConf().setAppName('TF-IDF Representation')
	sc = SparkContext(conf=conf)
	assert sc.version >= '1.5.1'

	sqlContext = SQLContext(sc)

	'''sbaronia - get 3 fields from json file and filter those with empty review'''
   	review = sqlContext.read.json(inputs).select('reviewText','overall','reviewTime').cache()
   	review_df = review.filter(review.reviewText != "").cache()

   	'''sbaronia - get year and rating and zip them with index'''
   	year_rdd = rdd_zip(review_df.map(extract_year).cache()).cache()
   	year_df = sqlContext.createDataFrame(year_rdd, ['year', 'index']).cache()

   	rating_rdd = rdd_zip(review_df.map(lambda line: float(line.overall)).cache()).cache()
   	rating_df = sqlContext.createDataFrame(rating_rdd, ['rating', 'index']).cache()
   	
	stop_words = stop_words_func(ntlk_path)

	'''sbaronia - rdd cotaining unique words from review'''
	clean_words_rdd = review_df.map(lambda review: clean_string_to_words(review.reviewText,stop_words)).filter(lambda x: x[0] != 'null').cache()

	'''sbaronia - finding tf-idf and zipping it with index'''
	tfidf_rdd = rdd_zip(tf_idf_cal(clean_words_rdd).cache()).cache()

	tfidf_df = sqlContext.createDataFrame(tfidf_rdd, ['tfidf', 'index']).cache()

	'''sbaronia - making dataframe with only rating and tfidf'''
	year_rating_df = rating_df.join(year_df, rating_df.index == year_df.index, 'outer').drop(rating_df.index).cache()
	tfyrrating_df = tfidf_df.join(year_rating_df, tfidf_df.index == year_rating_df.index, 'inner').drop(tfidf_df.index).cache()
	
	'''sbaronia - making training and testing rdd with <2014 and =2014 condition
	in a splitable format with :: '''
	train_rdd = tfyrrating_df.filter(tfyrrating_df.year < 2014) \
	                        .select('rating','tfidf') \
	                        .map(lambda line: (str(line.rating) + ' :: ' + str(line.tfidf))) \
	                        .coalesce(1) \
	                        .cache()
	
	test_rdd = tfyrrating_df.filter(tfyrrating_df.year == 2014) \
	                       .select('rating','tfidf') \
	                       .map(lambda line: (str(line.rating) + ' :: ' + str(line.tfidf))) \
	                       .coalesce(1) \
	                       .cache()
	
	'''sbaronia - save rdds to text''' 
   	train_rdd.saveAsTextFile(output + '/train-text')
	test_rdd.saveAsTextFile(output + '/test-text')

if __name__ == "__main__":
	main()


'''
To run on local -

spark-submit --master local[*] tf_idf_amazon_q1.py data/reviews_Pet_Supplies_p1.json output-p1

Output -

The training and testing data is stored in thr output directory
under directories train-text and test-text in the format

5.0 :: (1048576,[76143,320127,376269,403475,456494,494287,530684,639285,666067,777078,811757,830648,875187,1000429,1002337],[2.16791369297,5.49786078366,5.31418237455,4.90871726644,3.80672087979,3.56749119072,3.18132056481,12.2672465609,6.18087616529,2.38747323929,2.97400762406,6.78165002571,6.70387232467,4.2154406344,3.31821456101])
4.0 :: (1048576,[69189,98147,251465,605676,1000429,1001957,1035237],[2.52708083055,2.73242589877,2.90616897159,1.86786040341,1.40514687813,4.88858235803,3.92788801343])

where first number is review rating and second field is TF-IDF vector of review
'''