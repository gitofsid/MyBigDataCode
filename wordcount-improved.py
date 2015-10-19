"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * 
 * Problems -1. Wordcount by normalizing all the words
 *           2. Filter for any blank string
 *           3. Sort the words by frequency and alphabetically
 * ********************************************************************************************/

"""


from pyspark import SparkConf, SparkContext
import sys, operator
import unicodedata
import re, string

inputs = sys.argv[1]
output = sys.argv[2]

conf = SparkConf().setAppName('wordcount improved')
sc = SparkContext(conf=conf)

text = sc.textFile(inputs)

# sbaronia - get words after normalizing them and filter for 
# blank strings and conver them to lower case
wordsep = re.compile(r'[%s\s]+' % re.escape(string.punctuation))
words_fil = text.flatMap(lambda line: wordsep.split(line.lower())).filter(lambda w: len(w) > 0)
words = words_fil.map(lambda w: (unicodedata.normalize('NFD',w), 1))

# sbaronia - Use cache so when for first time it gets calculated
# it stays in memory to avoid recalculation
wordcount = words.reduceByKey(operator.add).coalesce(1).cache()

# sbaronia - output in a file with sort by words style
outdata_word = wordcount.sortBy(lambda (w,c): w).map(lambda (w,c): u"%s %i" % (w, c))
outdata_word.saveAsTextFile(output+'/by-word')

# sbaronia - output in a file with descending order of freqeuncy
outdata_fre = wordcount.sortBy(lambda (w,c): (-c,w)).map(lambda (w,c): u"%s %i" % (w, c))
outdata_fre.saveAsTextFile(output+'/by-freq')


