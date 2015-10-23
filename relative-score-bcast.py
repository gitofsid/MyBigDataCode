"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * Email - sbaronia@sfu.ca 
 * Student Number - 301288430
 * Assignment 3b, CMPT 732 
 * Date - 23 Oct 2015
 * 
 * Problems -1. Reuse reddit average code
 *           2. Add the avaerage of every unique subreddit to the JSON form
 *           3. Broadcast it to each executor for better memory usage
 *           4. A function that takes Broadcast variable and comment and 
 *              gives relative score
 * ********************************************************************************************/

"""

from pyspark import SparkConf, SparkContext
import sys,json

""" sbaronia - adding count and score for a subreddit """
def add_pairs(tuple1, tuple2):
	tuple_0 = tuple1[0]+tuple2[0]
	tuple_1 = tuple1[1]+tuple2[1]
	return (tuple_0, tuple_1)

""" sbaronia - calculate final average for every subreddit """
def reddit_average(tuple1):
	avg =  (1.0*tuple1[1])/(tuple1[0])
	return avg

""" sbaronia - this will calculate relative score """
def broadcast_join(bc_var, tuple_p):
	ravg = (1.0* tuple_p[1][0])/bc_var.value[tuple_p[0]]
	return (ravg,tuple_p[1][1])

inputs = sys.argv[1]
output = sys.argv[2]

conf = SparkConf().setAppName('relative score')
sc = SparkContext(conf=conf)

text = sc.textFile(inputs)

""" sbaronia - read json format """
js_python = text.map(json.loads).cache()

""" sbaronia - extract tags subreddit and score from every line """
subreddit_score = js_python.map(lambda line: (line['subreddit'], (1, line['score'])))
group_res = subreddit_score.reduceByKey(add_pairs).coalesce(1)

""" sbaronia - calculating average for every subreddit 
creating a dictionary out of it and then braodcasting it to executor """
average = group_res.mapValues(reddit_average).filter(lambda val: val[1]>0).coalesce(1)
bc_dic = dict(average.collect())
bc_var = sc.broadcast(bc_dic)

""" sbaronia - extracting subreddit, score, and author and adding average with it """
subreddit_new = js_python.map(lambda line: (line['subreddit'], (line['score'], line['author'])))

""" sbaronia - findind relative score and author and arranging in descending order """
subreddit_auth = subreddit_new.map(lambda var: (broadcast_join(bc_var,var))).sortBy(lambda (w,c): -w)

""" sbaronia - convert average to json format """
norm_average = subreddit_auth.map(json.dumps)

norm_average.saveAsTextFile(output)