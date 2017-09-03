package dataset;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

public class WingNut {

	final static String filename = "/cluster/WingNut.lrn";

	final static List<double[]> items;

	static {
		items = new ArrayList<double[]>();
		init(items);
	}

	static void init(List<double[]> items) {
		try (InputStream is = WholeSale.class.getResourceAsStream(filename)) {
			BufferedReader br = new BufferedReader(new InputStreamReader(is));
			String ln;
			while ((ln = br.readLine()) != null) {
				ln = ln.trim();
				if (ln.startsWith("%")) {
					continue;
				}
				String[] it = ln.split("\t");
				double[] r = new double[it.length];
				for (int i = 1; i < it.length; i++) {
					r[i] = Double.valueOf(it[i]);
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
			double[] v = items.get(i);
			ret[i] = MatrixUtils.createRealVector(new double[v.length]);
			for (int j = 0; j < v.length; j++) {
				ret[i].setEntry(j, v[j]);
			}
		}
		return ret;
	}

}
