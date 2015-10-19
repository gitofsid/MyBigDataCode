/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * 
 * Problems -1. Load log file into HBase Mapreduce
 *           2. input directory is first argument and table name is second   
 * ********************************************************************************************/


import java.io.IOException;
import java.text.ParseException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.mapreduce.TableMapReduceUtil;
import org.apache.hadoop.hbase.mapreduce.TableReducer;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;


public class LoadLogsMR extends Configured implements Tool {


	// sbaronia - we dont need a mapper but this implementation does the same
	// work as reading the lines from the file
	public static class LoadLogsMapper
	extends Mapper<LongWritable, Text, LongWritable, Text>{

		@Override
		public void map(LongWritable key, Text value, Context context
				) throws IOException, InterruptedException {

			context.write(key, value);
		}
	}
	
	// sbaronia - here we iterate over the lines we got and for every line
	// we create a put object that calls get_put method to put lines in
	// our table in raw and structured format
	public static class LoadLogsReducer
	extends TableReducer<LongWritable, Text, LongWritable> {

		@Override
		public void reduce(LongWritable key,  Iterable<Text> values, 
				Context context) throws IOException, InterruptedException {


			for (Text line : values) {
				Put put_value;
				try {
					put_value = LoadLogs.get_put(line.toString());
					context.write(key, put_value);
				} catch (ParseException e) {
					e.printStackTrace();
				}

			}
		}
	}

	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new LoadLogsMR(), args);
		System.exit(res);
	}

	@Override
	public int run(String[] args) throws Exception {
		Configuration config = HBaseConfiguration.create();
		Job job = Job.getInstance(config, "Loadlogs MR");

		// sbaronia - This block is all about reducer output and jar
		// arg[1] is the table to which data will be loaded - sbaronia-logs
		TableMapReduceUtil.addDependencyJars(job);
		TableMapReduceUtil.initTableReducerJob(args[1], LoadLogsReducer.class, job);
		job.setNumReduceTasks(3);

		job.setJarByClass(LoadLogsMR.class);

		job.setMapperClass(LoadLogsMapper.class);
		job.setReducerClass(LoadLogsReducer.class);
		
		// sbaronia - explicitly setting output key and value for mapper
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(Text.class);

		// sbaronia - input directory
		TextInputFormat.addInputPath(job, new Path(args[0]));

		return job.waitForCompletion(true) ? 0 : 1;
	}
}