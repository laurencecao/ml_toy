package dataset;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

public class WholeSale {

	final static String filename = "/wholesale/wholesale.csv";

	final static List<Integer[]> items;
	final static List<String> header;

	static {
		items = new ArrayList<Integer[]>();
		header = new ArrayList<String>();
		init(items, header);
	}

	static void init(List<Integer[]> items, List<String> header) {
		try (InputStream is = WholeSale.class.getResourceAsStream(filename)) {
			BufferedReader br = new BufferedReader(new InputStreamReader(is));
			String ln = br.readLine();
			header.addAll(Arrays.asList(ln.split(",")));
			while ((ln = br.readLine()) != null) {
				ln = ln.trim();
				String[] it = ln.split(",");
				Integer[] r = new Integer[it.length];
				for (int i = 0; i < it.length; i++) {
					r[i] = Integer.valueOf(it[i]);
				}
				items.add(r);
			}
			br.close();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public static RealVector[] getData() {
		RealVector[] ret = new RealVector[items.size()];
		for (int i = 0; i < items.size(); i++) {
			Integer[] v = items.get(i);
			ret[i] = MatrixUtils.createRealVector(new double[v.length]);
			for (int j = 0; j < v.length; j++) {
				ret[i].setEntry(j, v[j]);
			}
		}
		return ret;
	}

	public static String[] getHeader() {
		return header.toArray(new String[header.size()]);
	}

}
