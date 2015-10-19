/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * 
 * Problems -1. Parse JSON string and take out "subreddit" and "score" fields
 * 			 2. Write a mapper that outputs text and LongPairWritable. Text being
 * 				the name of subreddit and LongPairWritable being the pair of comment
 * 				count and score.
 *           3. Write a combiner that sums the comment counts and scores coming from mapper
 *           	for a given subreddit
 *           4. Write a reducer that generates average for each subreddit
 *           5. Use MultiLineJSONInputFormat as the input format
 * ********************************************************************************************/

import java.io.IOException;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

public class RedditAverage extends Configured implements Tool {

	// sbaronia - here the mapper will be reading files with JSON format
	// string and with the help of Jackson JSON parser two fields from 
	// input lines will be extracted - subreddit and score. A pair will
	// be made out of name of Subreddit and (comment count, score). 
	// Comment count will be set to 1 every time we parse a line. This
	// pair will then be written to context. LongPairWritable helper class
	// is used to implement (comment count, score) pair.
	public static class TokenizerMapper
	extends Mapper<LongWritable, Text, Text, LongPairWritable>{
		private LongPairWritable pair = new LongPairWritable();

		@Override
		public void map(LongWritable key, Text value, Context context
				) throws IOException, InterruptedException {
			ObjectMapper json_mapper = new ObjectMapper();

			JsonNode data = json_mapper.readValue(value.toString(), JsonNode.class);
			String reddit_text = data.get("subreddit").textValue();
			long reddit_score = data.get("score").longValue();

			pair.set(1, reddit_score);

			context.write(new Text(reddit_text), pair);
		}
	}

	// sbaronia - work of combiner here is to add the comment counts 
	// and scores respectively for a given subreddit. This also helps
	// reducer to find average without going over more data and also avoids 
	// sending large data over network.
	public static class RedditCombiner
	extends Reducer<Text, LongPairWritable, Text, LongPairWritable> {
		private LongPairWritable result = new LongPairWritable();

		@Override
		public void reduce(Text key, Iterable<LongPairWritable> values,
				Context context) throws IOException, InterruptedException {
			long sum_comcount = 0;
			long sum_score = 0;
			for (LongPairWritable pair : values) {
				sum_comcount += pair.get_0();
				sum_score += pair.get_1();
			}
			result.set(sum_comcount, sum_score);
			context.write(key, result);
		}
	}

	// sbaronia - work of the reducer here is to take mapper's key/value
	// output and calculate average of each subreddit. It writes Text,
	// DoubleWritable to file. Combiner assists reducer by providing sum
	// of comment and score for every subreddit.
	public static class RedditReducer
	extends Reducer<Text, LongPairWritable, Text, DoubleWritable> {
		private double result_average = 0;

		@Override
		public void reduce(Text key, Iterable<LongPairWritable> values,
				Context context) throws IOException, InterruptedException {
			long sum_comcount = 0;
			long sum_score = 0;
			for (LongPairWritable pair : values) {
				sum_comcount += pair.get_0();
				sum_score += pair.get_1();
			}
			// sbaronia - average in double
			result_average = (double)sum_score/(double)sum_comcount;

			context.write(key, new DoubleWritable(result_average));

		}
	}



	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new RedditAverage(), args);
		System.exit(res);
	}

	@Override
	public int run(String[] args) throws Exception {
		Configuration conf = this.getConf();
		Job job = Job.getInstance(conf, "reddit average");
		job.setJarByClass(RedditAverage.class);

		job.setInputFormatClass(MultiLineJSONInputFormat.class);

		job.setMapperClass(TokenizerMapper.class);
		job.setCombinerClass(RedditCombiner.class);
		job.setReducerClass(RedditReducer.class);

		job.setOutputKeyClass(Text.class);
		// sbaronia - output coming from mapper if of LongPairWritable type
		job.setOutputValueClass(LongPairWritable.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		TextInputFormat.addInputPath(job, new Path(args[0]));
		TextOutputFormat.setOutputPath(job, new Path(args[1]));

		return job.waitForCompletion(true) ? 0 : 1;
	}
}