package dataset;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

public class HomeLocation {

	// dataset:
	// homes in New York / homes in San Francisco ?
	// csv column
	// in_sf,beds,bath,price,year_built,sqft,price_per_sqft,elevation

	final static double[][] data;
	final static double[] label;
	static String[] header = null;
	static {
		List<double[]> d = init("/yoctol/part_1_data.csv");
		data = new double[d.size()][];
		label = new double[d.size()];
		for (int i = 0; i < d.size(); i++) {
			data[i] = Arrays.copyOfRange(d.get(i), 1, d.get(i).length);
			label[i] = d.get(i)[0];
		}
	}

	static RealVector[] getData() {
		RealVector[] ret = new RealVector[data.length];
		for (int i = 0; i < data.length; i++) {
			ret[i] = MatrixUtils.createRealVector(data[i]);
		}
		return ret;
	}

	static RealVector[] getLabel() {
		RealVector[] ret = new RealVector[label.length];
		for (int i = 0; i < label.length; i++) {
			ret[i] = MatrixUtils.createRealVector(new double[] { label[i] });
		}
		return ret;
	}

	static String[] getHeader() {
		return Arrays.copyOfRange(header, 1, header.length);
	}

	static List<double[]> init(String path) {
		try (InputStream fis = HomeLocation.class.getResourceAsStream(path)) {
			BufferedReader br = new BufferedReader(new InputStreamReader(fis));
			String ln = null;
			List<double[]> ret = new ArrayList<double[]>();
			while ((ln = br.readLine()) != null) {
				if (ln.startsWith("#")) {
					continue;
				}
				if (header == null) {
					header = ln.split(",");
					continue;
				}
				String[] n = ln.split(",");
				double[] r = new double[n.length];
				for (int i = 0; i < n.length; i++) {
					r[i] = Double.valueOf(n[i]);
				}
				ret.add(r);
			}
			br.close();
			return ret;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

}
