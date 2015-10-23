"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 *
 * Problems -1. Top 10000 frequent itemsets using FPGrowth
 *           2. Arrange them in freuency descending order
 *           3. For same frequencies arrange them in ascending order
 * ********************************************************************************************/

"""

from pyspark import SparkConf, SparkContext
from pyspark.mllib.fpm import FPGrowth
import sys, operator


inputs = sys.argv[1] #input
output = sys.argv[2] #output

conf = SparkConf().setAppName('frequent itemsets')
sc = SparkContext(conf=conf)

text = sc.textFile(inputs)

""" sbaronia - taking itemsets in int form and splitting then
of spaces, else ' ' becomes an itemset
"""
items = text.map(lambda line: map(int, line.strip().split(' ')))

""" sbaronia - calling FPGrowth function with support
0.0022 and partition 1, will give more than 10k frequent itemsets
"""
model = FPGrowth.train(items, 0.0022, 1)
fitems = model.freqItemsets()

""" sbaronia - here we sort every transaction in ascending order and 
then the entire 10k by descending order of frequencies and make 
and rdd from list of 10k items
"""
sort_transactions = sc.parallelize(fitems.map(lambda (i,c): (sorted(i), c)).sortBy(lambda (i,c): (-c,i)).take(10000))

sort_transactions.saveAsTextFile(output)
