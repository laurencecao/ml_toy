package dl.hmm;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.lang3.CharUtils;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

public class FourCoinProblemData {

	public final static String dataPath = "src/main/resources/hmm/fourcoin.txt";

	public static void main(String[] args) {
		int sample_size = 1000;
		int batch_size = 10;
		double a = 0.4d; // CoinA get HEAD/1 probs
		double b = 0.7d; // CoinB get HEAD/1 probs
		double c = 0.3d; // CoinA To CoinA probs
		double d = 0.6d; // CoinB To CoinA probs
		double e = 0.6d; // initial use CoinA probs
		String[][] data = generate(sample_size, batch_size, a, b, c, d, e);
		save(dataPath, data, batch_size, a, b, c, d, e);
	}

	static String[][] generate(int sample_size, int batch_size, double a, double b, double a2a, double b2a,
			double initial_a) {
		// 0. initialization: head probability of coin_a, coin_b, coin_a2b,
		// coin_b2a
		// 1. using 50% probability selection from coin_a or coin_b
		// 2. if coin_a then
		// 2a. trail coin_a and write output
		// 2b. trail coin_a2b and if head goto 2 else goto 3
		// 3. trail coin_b then
		// 3a. trail coin_b and write output
		// 3b. trail coin_b2a and if head goto 2 else goto 3
		// 4. repeat until sample_size finished
		String[][] ret = new String[2][];
		ret[0] = new String[sample_size];
		ret[1] = new String[sample_size];
		StringBuilder sb = new StringBuilder(sample_size);
		StringBuilder sb2 = new StringBuilder(sample_size);
		Random rnd = new Random(System.currentTimeMillis());
		double p;
		for (int i = 0; i < sample_size; i++) {
			sb.setLength(0);
			sb2.setLength(0);
			boolean useA = rnd.nextDouble() < initial_a;
			for (int j = 0; j < batch_size; j++) {
				// switch to coin A/B
				p = useA ? a : b;
				sb2.append(useA ? "A" : "B");
				// trail coin
				sb.append(rnd.nextDouble() < p ? "H" : "T");
				p = useA ? a2a : b2a;
				// selection coin
				useA = rnd.nextDouble() < p ? true : false;
			}
			ret[0][i] = sb.toString();
			ret[1][i] = sb2.toString();
		}
		return ret;
	}

	static void save(String path, String[][] data, int batch_size, double a, double b, double a2a, double b2a,
			double initial) {
		try (OutputStream os = new FileOutputStream(path)) {
			// add meta information to header
			StringBuilder sb = new StringBuilder();
			// sample_size, batch_size, initial_prob, CoinA_prob, CoinB_prob,
			// CoinA2CoinB_prob, CoinB2CoinA_prob
			sb.append("#\t").append(data.length).append("\t").append(batch_size).append("\t").append(initial)
					.append("\t");
			sb.append(a).append("\t").append(b).append("\t").append(a2a).append("\t").append(b2a);
			os.write(sb.toString().getBytes());
			os.write("\n".getBytes());
			for (int i = 0; i < data[0].length; i++) {
				sb.setLength(0);
				sb.append("#");
				for (int j = 0; j < batch_size; j++) {
					sb.append(data[1][i].charAt(j));
				}
				os.write(sb.toString().getBytes());
				os.write("\n".getBytes());
				sb.setLength(0);
				for (int j = 0; j < batch_size; j++) {
					sb.append(data[0][i].charAt(j));
				}
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
			String ln = br.readLine();
			String[] params = ln.split("\t");
			// int sample_size = Integer.valueOf(params[1]);
			int batch_size = Integer.valueOf(params[2].trim());
			List<RealVector> ret = new ArrayList<RealVector>();
			while ((ln = br.readLine()) != null) {
				if (!ln.startsWith("#")) {
					ln = ln.trim();
					RealVector d = MatrixUtils.createRealVector(new double[batch_size]);
					for (int i = 0; i < batch_size; i++) {
						double v = CharUtils.toChar(ln.charAt(i)) == 'H' ? 1d : 0d;
						d.setEntry(i, v);
					}
					ret.add(d);
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
