"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * 
 * Problems -1. Find shortest path from one node to another using Dijkstra Theorem
 *           2. Print intermediate paths
 * ********************************************************************************************/

"""

from pyspark import SparkConf, SparkContext
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql import SQLContext, DataFrame
from pyspark.sql.functions import split, min
import sys

def edge_pattern(line):
	lsplit = line.split(':')
	list = []
	
	if lsplit[1] == '':
		list.append((lsplit[0],lsplit[1]))
	else:
		for i in filter(None,lsplit[1].split(' ')):
			list.append((lsplit[0],i))
	return list

def found_paths(known_paths):
	nd = known_paths.select('Node', 'Distance') \
					.groupBy('Node') \
					.min('Distance') \
					.withColumnRenamed('min(Distance)','Distance') \
					.cache()
	nsd = known_paths.select('Node', 'Source', 'Distance').cache()
	
	clean_nds = nsd.join(nd, ['Node', 'Distance'], 'inner').dropDuplicates()
	
	return clean_nds

def final_path_print(fpath,src,dest):
	ns = fpath.select('Node','Source').map(lambda x: (int(x[0]),int(x[1]))).collect()

	dst_f = int(dest)
	src_f = int(src)
	list = []
	while (src_f not in list):
		k = 0
		for j in ns:
			if (j[0] == dst_f):
				list.append(j[0])
				dst_f = j[1]
				break
			else:
				k = k + 1
		if (k == len(ns)):
			break

	list.reverse()
	return list


def main():
	inputs = sys.argv[1]
	output = sys.argv[2]
	src = sys.argv[3]
	dest = sys.argv[4]

	conf = SparkConf().setAppName('shortest path')
	sc = SparkContext(conf=conf)
	assert sc.version >= '1.5.1'

	text = sc.textFile(inputs)
	sqlContext = SQLContext(sc)
	
	lines = text.flatMap(lambda line: edge_pattern(line))
	
	schema_edge = StructType([StructField('Src', StringType(), True),
						StructField('Destination', StringType(), True)])

	graph_edges = sqlContext.createDataFrame(lines, schema=schema_edge).cache()

	schema_path = StructType([StructField('Node', StringType(), True),
							  StructField('Source', StringType(), True),
						      StructField('Distance', IntegerType(), True)])

	
	first_node = sc.parallelize([(src,-1,0)])
	base_df = sqlContext.createDataFrame(first_node,schema=schema_path)
	base_join = base_df.join(graph_edges, graph_edges.Src == base_df.Node, 'inner')
	
	known_paths = base_join

	for i in range(6):

		clean_known_path = found_paths(known_paths).cache()
		clean_known_path.rdd \
						.map(lambda row1: 'Node: ' + str(row1.Node) + ' Source: ' + str(row1.Source) + ' Distance: ' + str(row1.Distance)) \
						.coalesce(1) \
						.saveAsTextFile(output + '/iter-' + str(i))

		freq_dest = clean_known_path.select('Node').rdd.filter(lambda x: map(int,x) == map(int,dest))
		if (freq_dest.count() == 1):
			break
	
		int_data = base_join.select('Destination', 'Node', (base_join.Distance+1).alias('Distance')).collect()
		int_df = sqlContext.createDataFrame(int_data, schema=schema_path)
		
		int_join = int_df.join(graph_edges, graph_edges.Src == int_df.Node, 'inner')
		
		base_join = int_join
		known_paths = known_paths.unionAll(int_join).cache()


	final_path = found_paths(known_paths)

	path_list = final_path_print(final_path,src,dest)
	vertical_list = []
	
	for val in path_list:
		vertical_list.append((val))
	
	final_rdd = sc.parallelize(vertical_list).coalesce(1)
	
	final_rdd.saveAsTextFile(output + '/path')
	
if __name__ == "__main__":
	main()



