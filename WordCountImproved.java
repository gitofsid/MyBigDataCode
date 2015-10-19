/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * 
 * Problems -1. Update mapper so it produces longwritable
 *           2. Use the pre-built reducer longsumreducer
 *           3. Update mapper to ignore punctuation and character casing for better couting
 *           4. Convert keys to NFD normal form
 * ********************************************************************************************/

// adapted from https://hadoop.apache.org/docs/current/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html

import java.io.IOException;
import java.util.Locale;
import java.text.BreakIterator;
import java.text.Normalizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.reduce.LongSumReducer;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class WordCountImproved extends Configured implements Tool {

	// sbaronia - this mapper will keep count in LongWritable format
	// to accommodate larger number (64 bit)
	public static class TokenizerMapper
	extends Mapper<LongWritable, Text, Text, LongWritable> {

		private final static LongWritable one = new LongWritable(1);
		private Text word = new Text();
		Locale locale = new Locale("en", "ca");
		BreakIterator breakiter = BreakIterator.getWordInstance(locale);


		// sbaronia - this map function will improve the counting by removing erroneous
		// punctuation and by changing words to lower case and normalizing it
		// for better word counting. 
		@Override
		public void map(LongWritable key, Text value, Context context
				) throws IOException, InterruptedException {
			breakiter.setText(value.toString());
			int start = breakiter.first();
			for (int end = breakiter.next();
					end != BreakIterator.DONE;
					start = end, end = breakiter.next()) {
				// sbaronia - this will convert string to lowercase and take the substring containing
				// word only and will trim off whitspaces and will normalize the result in NFD format
				word = new Text(Normalizer.normalize(value.toString().toLowerCase().substring(start,end).trim(), 
						Normalizer.Form.NFD));
				if ( word.toString().length() > 0 ) {
					context.write(word, one);
				}
			}		
		}
	}

	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new WordCountImproved(), args);
		System.exit(res);
	}

	@Override
	public int run(String[] args) throws Exception {
		Configuration conf = this.getConf();
		Job job = Job.getInstance(conf, "wordcount improved");
		job.setJarByClass(WordCountImproved.class);

		job.setInputFormatClass(TextInputFormat.class);

		job.setMapperClass(TokenizerMapper.class);
		//sbaronia - using pre-defined LongSumReducer as combiner and reducer
		job.setCombinerClass(LongSumReducer.class);
		job.setReducerClass(LongSumReducer.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(LongWritable.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		TextInputFormat.addInputPath(job, new Path(args[0]));
		TextOutputFormat.setOutputPath(job, new Path(args[1]));

		return job.waitForCompletion(true) ? 0 : 1;
	}
}