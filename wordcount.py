from pyspark import SparkConf, SparkContext
import sys, operator

inputs = sys.argv[1]
output = sys.argv[2]

conf = SparkConf().setAppName('word count')
sc = SparkContext(conf=conf)

text = sc.textFile(inputs)

words = text.flatMap(lambda line: line.split()).map(lambda w: (w, 1))

wordcount = words.reduceByKey(operator.add)

outdata = wordcount.sortBy(lambda (w,c): w).map(lambda (w,c): u"%s %i" % (w, c))
outdata.saveAsTextFile(output)