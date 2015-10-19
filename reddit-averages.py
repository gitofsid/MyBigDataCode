"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * 
 * Problems -1. Parse JSON string and take out "subreddit" and "score" fields
 *           2. Generates average for each subreddit
 * ********************************************************************************************/

"""


from pyspark import SparkConf, SparkContext
import sys, operator
import unicodedata
import re, string
import json

# sbaronia - add pairs of count and score
def add_pairs(tuple1, tuple2):
	tuple_0 = tuple1[0]+tuple2[0]
	tuple_1 = tuple1[1]+tuple2[1]
	return (tuple_0, tuple_1)

# sbaronia - calculate final average for every subreddit
def reddit_average(tuple1):
	avg =  (1.0*tuple1[1])/(tuple1[0])
	return avg

inputs = sys.argv[1]
output = sys.argv[2]

conf = SparkConf().setAppName('reddit average')
sc = SparkContext(conf=conf)

text = sc.textFile(inputs)

# sbaronia - read json format 
js_python = text.map(json.loads)

# sbaronia - extract tags subreddit and score from every line
subreddit_score = js_python.map(lambda line: (line['subreddit'], (1, line['score'])))
group_res = subreddit_score.reduceByKey(add_pairs).coalesce(1)

average = group_res.mapValues(reddit_average).coalesce(1)

# sbaronia - convert average to json format
norm_average = average.map(json.dumps)

norm_average.saveAsTextFile(output)

