"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * Email - sbaronia@sfu.ca 
 * Student Number - 301288430
 * Assignment 3b, CMPT 732 
 * Date - 23 Oct 2015
 * 
 * Problems -1. Read from JSON format to find host and bytes values
 *           2. Sum the values of bytes and count for each host
 *           3. Implement a second mapper to calculate value of r - correlation coeff using 
 *              a different formula using average values of count and bytes
 * ********************************************************************************************/

"""


from pyspark import SparkConf, SparkContext
import sys, re, math

""" sbaronia - this function adds the value of bytes and count 
for a given host.
"""
def add_tuples(a,b):
	return tuple(sum(p) for p in zip(a,b))

""" sbaronia - this function calculates value of 
mean for counts and bytes for every unique host
"""
def mean_elements(tuple_p):
	n = sum(1 for i,j in tuple_p)
	avg_x = sum(i for i,j in tuple_p)/n
	avg_y = sum(j for i,j in tuple_p)/n
	return n,avg_x,avg_y

""" sbaronia - this function calculates the elements needed 
for finding r and r2
"""
def calculations(tuple_q,avgx,avgy):
	Sx2 = sum((i-avgx)*(i-avgx) for i,j in tuple_q)
	Sy2 = sum((j-avgy)*(j-avgy) for i,j in tuple_q)
	Sxy = sum((i-avgx)*(j-avgy) for i,j in tuple_q)
	return Sx2,Sy2,Sxy


inputs = sys.argv[1] #input
output = sys.argv[2] #output

conf = SparkConf().setAppName('correlate logs')
sc = SparkContext(conf=conf)

text = sc.textFile(inputs)


linere = re.compile("^(\\S+) - - \\[(\\S+) [+-]\\d+\\] \"[A-Z]+ (\\S+) HTTP/\\d\\.\\d\" \\d+ (\\d+)$")
line_match = text.map(lambda line: linere.match(line)).filter(None)

""" sbaronia - extract host name and bytes """
pair = line_match.map(lambda line: (line.group(1), (1, int(line.group(4)))))

""" sbaronia - reduce by key which is host to add
all the counts of request and bytes, then we groupit 
by key so it can be iterated over and calculation gives 
all the necessary values we need
"""
same_key = pair.reduceByKey(add_tuples).filter(None).coalesce(1)
group = same_key.map(lambda same_key: (1, same_key[1]))
group_key = group.groupByKey().cache()

""" sbaronia - calling function to calculate mean and 
collecting values of n and averages 
"""
mean_el = group_key.map(lambda val: mean_elements(val[1]))
mean_el_val = mean_el.collect()

n = mean_el_val[0][0]
avg_x = mean_el_val[0][1]
avg_y = mean_el_val[0][2]

""" sbaronia - calling calculations once we have averages values
and then get sum of terms, and values to calculate r
"""
cal_el = group_key.map(lambda val: calculations(val[1], avg_x, avg_y))
cal_el_val = cal_el.collect()

Sx2 = cal_el_val[0][0]
Sy2 = cal_el_val[0][1]
Sxy = cal_el_val[0][2]


sum_num = Sxy
sum_den = math.sqrt(Sx2)*math.sqrt(Sy2)

""" sbaronia - if sum of denominator is not 0 then find r and r2 """
if sum_den != 0:
	r = sum_num/sum_den
	r2 = r*r

""" sbaronia - making an RDD from different values"""
result = sc.parallelize(['n %i' % n, 'avg_x %i' % avg_x, 'avg_y %i' % avg_y, 'Sx2 %i' % Sx2, 'Sy2 %i' % Sy2, 'Sxy %i' % Sxy, 'r %0.8f' % r, 'r2 %0.8f' % r2 ])

result.saveAsTextFile(output)
