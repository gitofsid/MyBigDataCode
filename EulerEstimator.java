/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * 
 * Problems -1. Use java.util.Random to generate random values and estimate e
 *           2. Use hadoop counter to count total number of iterations
 *           3. Set output format NullOutputFormat
 *           4. Use hadoop counter to count random numbers generated
 *           5. Divide count by iterations to get value of Euler's constant e
 * ********************************************************************************************/



import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.NullOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;



public class EulerEstimator extends Configured implements Tool {


	public static class EulerMapper
	extends Mapper<LongWritable, Text, NullOutputFormat, NullOutputFormat>{


		@Override
		public void map(LongWritable key, Text value, Context context
				) throws IOException, InterruptedException {
			// sbaronia - parsing iterations from lines in file and
			// keeping global count of values generated
			long iterations = Integer.parseInt(value.toString());
			long count = 0;
			
			// sbaronia - using hash from file name and offset from beginning of line
			// to generate unique random values
			String filename = ((FileSplit) context.getInputSplit()).getPath().getName();
			long hash = filename.hashCode();
			Random random_gen = new Random(hash*key.get());
			
			long i = 0;

			while ( i < iterations) {
				double sum = 0.0;
				while (sum < 1) {
					// sbaronia - summing random values between 0 and 1 till 
					// sum gets greater than 1
					sum += random_gen.nextDouble();
					count++;
				}
				i++;
			}
			
			// sbaronia - using hadoop counter to count total number of iterations
			// and total number of random numbers generated
			context.getCounter("Euler", "iterations").increment(iterations);
			context.getCounter("Euler", "count").increment(count);

		}
	}


	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new EulerEstimator(), args);
		System.exit(res);
	}

	@Override
	public int run(String[] args) throws Exception {
		Configuration conf = this.getConf();
		Job job = Job.getInstance(conf, "euler estimator");
		job.setJarByClass(EulerEstimator.class);

		job.setInputFormatClass(TextInputFormat.class);

		job.setMapperClass(EulerMapper.class);

		// sbaronia - setting outformat as NullOutputFormat
		job.setOutputFormatClass(NullOutputFormat.class);
		// sbaronia - we are only taking input path argument as output will be displayed on console 
		TextInputFormat.addInputPath(job, new Path(args[0]));

		return job.waitForCompletion(true) ? 0 : 1;
	}
}