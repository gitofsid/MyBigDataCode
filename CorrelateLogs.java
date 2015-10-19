/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * 
 * Problems -1. Read from table's struct column to find host and bytes values
 *           2. Sum the values of bytes and count for each host
 *           3. Implement a second mapper to calculate value of r - correlation coeff
 * ********************************************************************************************/


import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hbase.Cell;
import org.apache.hadoop.hbase.CellUtil;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.client.Scan;
import org.apache.hadoop.hbase.io.ImmutableBytesWritable;
import org.apache.hadoop.hbase.mapreduce.TableMapReduceUtil;
import org.apache.hadoop.hbase.mapreduce.TableMapper;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.chain.ChainReducer;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;


public class CorrelateLogs extends Configured implements Tool {

	public static class CorrelateLogsMapper
	extends TableMapper<Text, LongPairWritable> {

		@Override
		public void map(ImmutableBytesWritable key, Result result, Context context
				) throws IOException, InterruptedException {

			LongPairWritable pair = new LongPairWritable();
			byte[] cell_hbytes = new byte [] {'0'};
			byte[] cell_bbytes = new byte[] {0};


			Cell cell_host = result.getColumnLatestCell(Bytes.toBytes("struct"), Bytes.toBytes("host"));
			Cell cell_bytes = result.getColumnLatestCell(Bytes.toBytes("struct"), Bytes.toBytes("bytes"));
			if (cell_host != null && cell_bytes != null) {
				cell_hbytes  = CellUtil.cloneValue(cell_host);
				cell_bbytes = CellUtil.cloneValue(cell_bytes);
				pair.set(1,Bytes.toLong(cell_bbytes));
			} 
			
			context.write(new Text(Bytes.toString(cell_hbytes)), pair);
		}
	}

	public static class CorrelateLogsReducer
	extends Reducer<Text, LongPairWritable, Text, LongPairWritable> {
		private LongPairWritable result = new LongPairWritable();

		@Override
		public void reduce(Text key, Iterable<LongPairWritable> values, 
				Context context) throws IOException, InterruptedException {
			long sum_request = 0;
			long sum_bytes = 0;

			for (LongPairWritable pair : values) {
				sum_request += pair.get_0();
				sum_bytes += pair.get_1();
			}

			result.set(sum_request,  sum_bytes);
			context.write(key, result);	
		}
	}

	// sbaronia - second mapper which will take values from reducer and will
	// calculate values of sums and then finally value of r
	public static class CorrelateLogsMapper2
	extends Mapper<Text, LongPairWritable, Text, DoubleWritable> {
		String [] outputs = {"n", "Sx", "Sx2", "Sy", "Sy2", "Sxy", "r", "r2"};
		long n, Sx, Sx2, Sy, Sy2, Sxy;
		double r;

		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			n = Sx = Sx2 = Sy = Sy2 = Sxy = 0L;
			r = 0;
		}
		
		// sbaronia - here we are calculating all sums 
		@Override
		public void map(Text key, LongPairWritable pair, 
				Context context) throws IOException, InterruptedException {
			
			n++;
			Sx += pair.get_0();
			Sy += pair.get_1();
			Sx2 += pair.get_0()*pair.get_0();
			Sy2 += pair.get_1()*pair.get_1();
			Sxy += pair.get_0()*pair.get_1();
			
		}
		
		// sbaronia - finding the value of r by using above sums
		@Override
		public void cleanup(Context context) throws IOException, InterruptedException {

			double d_n = n;
			double d_Sx = Sx;
			double d_Sy = Sy;
			double d_Sx2 = Sx2;
			double d_Sy2 = Sy2;
			double d_Sxy = Sxy;
			
			double num =  (d_n*d_Sxy - d_Sx*d_Sy);
			double den = ((Math.sqrt(d_n*d_Sx2 - d_Sx*d_Sx)*Math.sqrt(d_n*d_Sy2 - d_Sy*d_Sy)));

			r = num/den;
			
			context.write(new Text(outputs[0]), new DoubleWritable(n));
			context.write(new Text(outputs[1]), new DoubleWritable(Sx));
			context.write(new Text(outputs[2]), new DoubleWritable(Sx2));
			context.write(new Text(outputs[3]), new DoubleWritable(Sy));
			context.write(new Text(outputs[4]), new DoubleWritable(Sy2));
			context.write(new Text(outputs[5]), new DoubleWritable(Sxy));
			context.write(new Text(outputs[6]), new DoubleWritable(r));
			context.write(new Text(outputs[7]), new DoubleWritable(r*r));	
		}
	}

	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new CorrelateLogs(), args);
		System.exit(res);
	}

	@Override
	public int run(String[] args) throws Exception {
		Configuration config = this.getConf();
		Job job = Job.getInstance(config, "Correlated Logs");
		Scan scan = new Scan();
		// sbaronia - only taking struct family
		scan.addFamily(Bytes.toBytes("struct"));

		job.setJarByClass(CorrelateLogs.class);
		
		// sbaronia - setting tablemapper with name of table as input
		TableMapReduceUtil.addDependencyJars(job);
		TableMapReduceUtil.initTableMapperJob(args[0], scan, 
				CorrelateLogsMapper.class, Text.class, LongPairWritable.class, job);

		job.setNumReduceTasks(1);

		job.setMapperClass(CorrelateLogsMapper.class);
		job.setReducerClass(CorrelateLogsReducer.class);

		// sbaronia - output directory for storing value of r and other vars
		TextOutputFormat.setOutputPath(job, new Path(args[1]));
		
		
		// sbaronia - adding a chain mapper and setting old reducer as reducer
		ChainReducer.setReducer(job, CorrelateLogsReducer.class, Text.class, LongPairWritable.class,
			Text.class, LongPairWritable.class, new Configuration (false));
		ChainReducer.addMapper(job, CorrelateLogsMapper2.class, Text.class, LongPairWritable.class, 
			Text.class, DoubleWritable.class, new Configuration(false));

		return job.waitForCompletion(true) ? 0 : 1;
	}
}