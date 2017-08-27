package dataset;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class TSPCitys {

	final static String dataName = "/tsp/dantzig42_d.txt";

	final static RealMatrix cityCost;

	static {
		cityCost = init();
	}

	static RealMatrix init() {
		RealMatrix ret = null;
		try (InputStream fis = TSPCitys.class.getResourceAsStream(dataName)) {
			BufferedReader br = new BufferedReader(new InputStreamReader(fis));
			String line = null;
			List<double[]> data = new ArrayList<double[]>();
			while ((line = br.readLine()) != null) {
				line = line.trim().replaceAll(" +", " ");
				String[] items = line.split(" ");
				double[] d = new double[items.length];
				for (int i = 0; i < items.length; i++) {
					d[i] = Double.valueOf(items[i]);
				}
				data.add(d);
			}
			double[][] m = new double[data.size()][];
			for (int i = 0; i < m.length; i++) {
				m[i] = data.get(i);
			}
			ret = MatrixUtils.createRealMatrix(m);
			br.close();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		return ret;
	}

}
