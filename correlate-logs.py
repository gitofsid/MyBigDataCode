"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * 
 * Problems -1. Read from JSON format to find host and bytes values
 *           2. Sum the values of bytes and count for each host
 *           3. Implement a second mapper to calculate value of r - correlation coeff
 * ********************************************************************************************/

"""


from pyspark import SparkConf, SparkContext
import sys, re, math

""" sbaronia - this function adds the value of bytes and count 
for a given host.
"""
def add_tuples(a,b):
	return tuple(sum(p) for p in zip(a,b))

""" sbaronia - here we calculate the values needed for
calculation of r, like n, Sx, Sx2, Sy, Sy2, Sxy
"""
def calculation(tuple_p):
	n = sum(1 for i,j in tuple_p)
	Sx = sum(i for i,j in tuple_p)
	Sy = sum(j for i,j in tuple_p)
	Sx2 = sum(i*i for i,j in tuple_p)
	Sy2 = sum(j*j for i,j in tuple_p)
	Sxy = sum(i*j for i,j in tuple_p)
	return n,Sx,Sy,Sx2,Sy2,Sxy


inputs = sys.argv[1] #input
output = sys.argv[2] #output

conf = SparkConf().setAppName('correlate logs')
sc = SparkContext(conf=conf)

text = sc.textFile(inputs)


linere = re.compile("^(\\S+) - - \\[(\\S+) [+-]\\d+\\] \"[A-Z]+ (\\S+) HTTP/\\d\\.\\d\" \\d+ (\\d+)$")
line_match = text.map(lambda line: linere.match(line)).filter(None)

""" sbaronia - extracting host, number of bytes and 
assigning 1 to every reuest as a count
"""
pair = line_match.map(lambda line: (line.group(1), (1, int(line.group(4)))))

""" sbaronia - reduce by key which is host to add
all the counts of request and bytes, then we groupit 
by key so it can be iterated over and calculation gives 
all the necessary values we need
"""
same_key = pair.reduceByKey(add_tuples).filter(None).coalesce(1)
group = same_key.map(lambda same_key: (1, same_key[1]))
group_key = group.groupByKey()

sum_pair_2 = group_key.map(lambda val: calculation(val[1]))
sum_pair = sum_pair_2.collect()

""" sbaronia - values returned from calculation are 
assigned to correct variable
"""
n   = sum_pair[0][0]
Sx  = sum_pair[0][1]
Sy  = sum_pair[0][2]
Sx2 = sum_pair[0][3]
Sy2 = sum_pair[0][4]
Sxy = sum_pair[0][5]


sum_num = n*Sxy - Sx*Sy
sum_den = math.sqrt(n*Sx2 - Sx*Sx)*math.sqrt(n*Sy2 - Sy*Sy)

""" sbaronia - if denominator is not 0 then find r and r2
"""
if sum_den != 0:
	r = sum_num/sum_den
	r2 = r*r

""" sbaronia - make a new RDD and print all values
"""
result = sc.parallelize(['n %i' % n, 'Sx %i' % Sx, 'Sy %i' % Sy, 'Sx2 %i' % Sx2, 'Sy2 %i' % Sy2, 'Sxy %i' % Sxy, 'r %0.8f' % r, 'r2 %0.8f' % r2 ])

result.saveAsTextFile(output)
