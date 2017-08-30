package dl.em;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

public class ThreeCoinProblemData {

	public final static String dataPath = "src/main/resources/em/threecoin.txt";

	public static void main(String[] args) {
		int sample_size = 10000;
		int batch_size = 10;
		double a = 0.3d;
		double b = 0.6d;
		double c = 0.7d;
		RealVector[] data = generate(sample_size, batch_size, a, b, c);
		save(dataPath, data, a, b, c);
	}

	static RealVector[] generate(int sample_size, int batch_size, double a, double b, double c) {
		// 0. initialization: head probability of coin_a, coin_b, coin_c
		// 1. trail coin_c, get head or tail
		// 2. trail coin_a if head loop for batch_size
		// 3. trail coin_b if head loop for batch_size
		// 4. repeat until sample_size finished
		RealVector[] ret = new RealVector[sample_size];
		ThreadLocalRandom rnd = ThreadLocalRandom.current();
		for (int i = 0; i < sample_size; i++) {
			double p;
			ret[i] = MatrixUtils.createRealVector(new double[batch_size]);
			if (rnd.nextDouble() < c) {
				p = a;
			} else {
				p = b;
			}
			for (int j = 0; j < batch_size; j++) {
				ret[i].setEntry(j, rnd.nextDouble() < p ? 1.0d : 0d);
			}
		}
		return ret;
	}

	static void save(String path, RealVector[] data, double a, double b, double c) {
		try (OutputStream os = new FileOutputStream(path)) {
			int batch_size = data[0].getDimension();
			// add meta information to header
			StringBuilder sb = new StringBuilder();
			sb.append("#\t").append(data.length).append("\t").append(batch_size).append("\t");
			sb.append(a).append("\t").append(b).append("\t").append(c);
			os.write(sb.toString().getBytes());
			os.write("\n".getBytes());
			for (int i = 0; i < data.length; i++) {
				sb.setLength(0);
				for (int j = 0; j < batch_size; j++) {
					sb.append(Double.valueOf(data[i].getEntry(j)).intValue()).append("\t");
				}
				sb.deleteCharAt(sb.length() - 1);
				os.write(sb.toString().getBytes());
				os.write("\n".getBytes());
			}
		} catch (Exception ex) {
			throw new RuntimeException(ex);
		}
	}

	public static RealVector[] load(String path) {
		try (InputStream is = new FileInputStream(path)) {
			BufferedReader br = new BufferedReader(new InputStreamReader(is));
			String ln = null;
			List<RealVector> ret = new ArrayList<RealVector>();
			while ((ln = br.readLine()) != null) {
				if (!ln.startsWith("#")) {
					ret.add(readline(ln));
				}
			}
			return ret.toArray(new RealVector[ret.size()]);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	static RealVector readline(String ln) {
		String[] items = ln.trim().split("\t");
		double[] ret = new double[items.length];
		for (int i = 0; i < items.length; i++) {
			ret[i] = Double.valueOf(items[i]);
		}
		return MatrixUtils.createRealVector(ret);
	}

}
