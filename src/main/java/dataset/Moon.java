package dataset;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class Moon {

	final static String MOON_PATH = "/moon.csv";

	static List<double[]> _data = new ArrayList<double[]>();
	static List<Integer> labels = new ArrayList<Integer>();

	static {
		init();
	}

	static void init() {
		try (InputStream is = WholeSale.class.getResourceAsStream(MOON_PATH)) {
			BufferedReader br = new BufferedReader(new InputStreamReader(is));
			String ln = null; //
			while ((ln = br.readLine()) != null) {
				ln = ln.trim();
				if (ln.startsWith("%") || ln.startsWith("#")) {
					continue;
				}
				String[] it = ln.split(",");
				double[] r = new double[it.length - 1];
				for (int i = 0; i < it.length - 1; i++) {
					r[i] = Double.valueOf(it[i]);
				}
				int l = Integer.valueOf(it[it.length - 1]);
				_data.add(r);
				labels.add(l);
			}
			br.close();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public static double[][] getData() {
		double[][] ret = new double[_data.size()][];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = _data.get(i);
		}
		return ret;
	}

	public static int[] getLabel() {
		int[] ret = new int[labels.size()];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = labels.get(i);
		}
		return ret;
	}

}
