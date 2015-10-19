/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * 
 * Problems -1. Create an inputformat that reads multi-lines JSON file
 * 			 2. Change nextKeyValue function so multi-lined JSON can be read to one line
 * 				standard format. 
 * 			 3. Also make sure that standard single lined JSON functionality does not break.
 * 				Basically make this work for multiline as well as single line input.
 * ********************************************************************************************/

import java.io.IOException;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.LineRecordReader;

import com.google.common.base.Charsets;

public class MultiLineJSONInputFormat extends TextInputFormat {

	public class MultiLineRecordReader extends RecordReader<LongWritable, Text> {
		LineRecordReader linereader;
		LongWritable current_key;
		Text current_value;

		public MultiLineRecordReader(byte[] recordDelimiterBytes) {
			linereader = new LineRecordReader(recordDelimiterBytes);
		}

		@Override
		public void initialize(InputSplit genericSplit,
				TaskAttemptContext context) throws IOException {
			linereader.initialize(genericSplit, context);
		}

		@Override
		public boolean nextKeyValue() throws IOException {
			// sbaronia - read an initial key ad value to start with
			Text current_value_int = new Text();
			boolean res = linereader.nextKeyValue();
			current_key = linereader.getCurrentKey();
			current_value = linereader.getCurrentValue();
			
			// sbaronia - this takes care of single lined standard JSON input. Here I am
			// getting current line and only checking the last character of it to see if
			// its }, which means single liend JSON, so return from here
			if (current_value != null && 
					current_value.toString().substring(current_value.getLength()-1).trim().equals("}")) {
				return res;

			}
			
			// sbaronia - here we check multi-lined JSON input line by line till 
			// we reach }. The purpose is to convert into standard single line format.
			// As we read lines we keep appending to make a single line.
			while (current_value != null && !current_value.toString().trim().equals("}")  ) {
				if (current_value_int != null) {
					current_value_int.append(current_value.copyBytes(),
							0, current_value.getLength());
				}
				res = linereader.nextKeyValue();
			}
			
			//  sbaronia - we are here as } has been detected. Now append that at the end
			// and set current value to this one single line JSON and then return.
			if (current_value != null) {
				current_value_int.append(current_value.copyBytes(),0, current_value.getLength());
				current_value.set(current_value_int);
			}

			return res;
		}

		@Override
		public float getProgress() throws IOException {
			return linereader.getProgress();
		}

		@Override
		public LongWritable getCurrentKey() {
			return current_key;
		}

		@Override
		public Text getCurrentValue() {
			return current_value;
		}

		@Override
		public synchronized void close() throws IOException {
			linereader.close();
		}
	}

	// shouldn't have to change below here

	@Override
	public RecordReader<LongWritable, Text> 
	createRecordReader(InputSplit split,
			TaskAttemptContext context) {
		// same as TextInputFormat constructor, except return MultiLineRecordReader
		String delimiter = context.getConfiguration().get(
				"textinputformat.record.delimiter");
		byte[] recordDelimiterBytes = null;
		if (null != delimiter)
			recordDelimiterBytes = delimiter.getBytes(Charsets.UTF_8);
		return new MultiLineRecordReader(recordDelimiterBytes);
	}

	@Override
	protected boolean isSplitable(JobContext context, Path file) {
		// let's not worry about where to split within a file
		return false;
	}
}