/* ******************************************************************************************
 * Name - Siddharth Baronia 
 * 
 * Problems -1. Using hbase shell create table sbaronia-logs with two columns raw and struct
 *           2. Store the lines from text file in their raw format into one column
 *           3. Parse data like - host, date, path, bytes from those lines and
 *              store in second column called struct
 *           4. Use hadoop counter to count random numbers generated
 *           5. Divide count by iterations to get value of Euler's constant e
 * ********************************************************************************************/



import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.codec.digest.DigestUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.HTableDescriptor;
import org.apache.hadoop.hbase.TableName;
import org.apache.hadoop.hbase.client.Connection;
import org.apache.hadoop.hbase.client.ConnectionFactory;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Table;
import org.apache.hadoop.hbase.util.Bytes;

public class LoadLogs {

	// sbaronia - get_put function that creates row key as MD5 hash of line
	// extracts the useful patterns from line, get date, host, path and bytes
	// writes raw line to raw column family's line column and host, data, path, bytes
	// to corresponding columns of struct column family. And return the value.
	public static Put get_put(String line) throws ParseException {
		
		// sbaronia - storing date and bytes in Long with 0L values
		// the host and path are in string with N/A default value which 
		// will be used if patter doesn't match
		Long ldate = 0L, bytes = 0L;
		String host = new String("N/A");
		String path = new String("N/A");

		byte[] rowkey = DigestUtils.md5(line);
		final Pattern patt = Pattern.compile("^(\\S+) - - \\[(\\S+) [+-]\\d+\\] \"[A-Z]+ (\\S+) HTTP/\\d\\.\\d\" \\d+ (\\d+)$");
		Matcher matcher = patt.matcher(line);

		// sbaronia - found four groups by matching the pattern
		if (matcher.matches()) {
			host = matcher.group(1).toString();

			SimpleDateFormat dateparse = new SimpleDateFormat("dd/MMM/yyyy:HH:mm:ss");
			Date date = dateparse.parse(matcher.group(2).toString());
			ldate = date.getTime();

			path = matcher.group(3).toString();

			bytes = Long.valueOf(matcher.group(4).toString());		
		}

		Put value = new Put (rowkey);
		
		// sbaronia - add lines and values to our columns
		value.addColumn(Bytes.toBytes("raw"), Bytes.toBytes("line"), Bytes.toBytes(line));
		value.addColumn(Bytes.toBytes("struct"), Bytes.toBytes("host"), Bytes.toBytes(host));
		value.addColumn(Bytes.toBytes("struct"), Bytes.toBytes("date"), Bytes.toBytes(ldate));
		value.addColumn(Bytes.toBytes("struct"), Bytes.toBytes("path"), Bytes.toBytes(path));
		value.addColumn(Bytes.toBytes("struct"), Bytes.toBytes("bytes"), Bytes.toBytes(bytes));


		return value;

	}


	public static void main(String[] args) throws Exception {
		
		// sbaronia - create a config, a connection and get the table 
		// from name passed as first argument - sbaronia-logs
		Configuration config = HBaseConfiguration.create();
		Connection connection = ConnectionFactory.createConnection(config);
		HTableDescriptor tdesc = new HTableDescriptor(TableName.valueOf(args[0]));
		Table table = connection.getTable(tdesc.getTableName());
		
		// sbaronia - open a file passed as second argument and a bufferreader
		File myFile = new File(args[1]);
		BufferedReader myBuffer = new BufferedReader (new FileReader(myFile));
		
		// sbaronia - a try and finally block where get_put is called 
		// and table is written with the values set through function call
		// and at the end buffer, table and connection are closed
		try {
			String line;
			while ((line = myBuffer.readLine()) != null) {
				Put value = get_put(line);
				table.put(value);
			}
		} finally {
			myBuffer.close();
			table.close();
			connection.close();
		}			
	}
}