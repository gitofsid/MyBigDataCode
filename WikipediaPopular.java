/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * 
 * Problems -1. Find number of times most visited page was visited each hour
 *           2. Report only english pages
 *           3. Dont consider Main_page count and title starting with Special
 *           4. Implement combiner and reducer
 * ********************************************************************************************/

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;


public class WikipediaPopular extends Configured implements Tool {

	public static class TokenizerMapper
	extends Mapper<LongWritable, Text, Text, LongWritable>{

		@Override
		public void map(LongWritable key, Text value, Context context
				) throws IOException, InterruptedException {
			int index = 0;
			StringTokenizer itr = new StringTokenizer(value.toString());
			// sbaronia - splitting read line and storing elements in array
			String[] linesplitdata = new String[itr.countTokens()];
			while(itr.hasMoreTokens()) {
				linesplitdata[index] = itr.nextToken();
				++index;
			}

			// sbaronia - if first element of array is en and does not contain Main_Page
			// and does not start with Special: then write part of name of file and
			// frequency of visits to context
			if (linesplitdata[0].equals("en") && !linesplitdata[1].contains("Main_Page") 
					&& !linesplitdata[1].startsWith("Special:") ) {
				String filename = ((FileSplit) context.getInputSplit()).getPath().getName();
				// sbaronia - from file name pagecounts-20141201-000000.gz it gets substring 
				// from occurence of first '-' to '.' - 20141201-000000 and further we get a 
				// substring till 201401201-00
				String filename_sub = StringUtils.substringBetween(filename, "-", ".");
				String filename_final = filename_sub.substring(0, filename_sub.length()-4);
				context.write(new Text(filename_final), new LongWritable(Long.valueOf(linesplitdata[2]).longValue()));
			}
		}
	}

	// sbaronia - work of combiner here is to go through the list of 
	// all english entries comning from mapper for an hourly file and 
	// find the highest, which reduces the load on reduer. In case 
	// when a file gets splitted into multiple mapper for processing, 
	// this combiner will be helpful. This also helps in avoiding large 
	// data to be sent over network. 
	public static class WikiPediaCombiner
	extends Reducer<Text, LongWritable, Text, LongWritable> {
		private LongWritable result_comcount = new LongWritable(0);

		@Override
		public void reduce(Text key, Iterable<LongWritable> values,
				Context context) throws IOException, InterruptedException {
			long max_comcount = 0;

			for (LongWritable count : values) {
				if (count.get() > max_comcount) {
					max_comcount = count.get();
				}
			}
			result_comcount.set(max_comcount);
			context.write(key, result_comcount);			
		}
	}

	// sbaronia - this reducer is similar to the combiner above.
	// Here again we find the highest entry for a file in case results
	// for a same file is coming in parts
	public static class WikiPediaReducer
	extends Reducer<Text, LongWritable, Text, LongWritable> {
		private LongWritable result_count = new LongWritable(0);

		@Override
		public void reduce(Text key, Iterable<LongWritable> values,
				Context context) throws IOException, InterruptedException {
			long max_count = 0;

			for (LongWritable count : values) {
				if (count.get() > max_count) {
					max_count = count.get();
				}
			}
			result_count.set(max_count);
			context.write(key, result_count);			
		}
	}

	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new WikipediaPopular(), args);
		System.exit(res);
	}

	@Override
	public int run(String[] args) throws Exception {
		Configuration conf = this.getConf();
		Job job = Job.getInstance(conf, "wikipedia popular");
		job.setJarByClass(WikipediaPopular.class);

		job.setInputFormatClass(TextInputFormat.class);

		job.setMapperClass(TokenizerMapper.class);
		job.setCombinerClass(WikiPediaCombiner.class);
		job.setReducerClass(WikiPediaReducer.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(LongWritable.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		TextInputFormat.addInputPath(job, new Path(args[0]));
		TextOutputFormat.setOutputPath(job, new Path(args[1]));

		return job.waitForCompletion(true) ? 0 : 1;
	}
}
