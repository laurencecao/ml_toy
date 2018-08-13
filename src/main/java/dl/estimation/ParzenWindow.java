package dl.estimation;

import java.io.IOException;

import org.apache.commons.math3.distribution.NormalDistribution;

import utils.JFreeChartUtils;
import utils.MatrixUtil;

public class ParzenWindow {

	public static void main(String[] args) throws IOException {
		int sz = 50000;
		int bin = 100;
		double center = 10d;
		double var = 3d;
		double[] samples = getRealDist(sz, center, var);
		System.out.println(MatrixUtil.pretty(samples));
		System.out.println("------------------------------------");
		double distance = center * 1.5;
		double[][] p = ParzenKDE(bin, new double[] { center - distance, center + distance }, samples);
		for (int i = 0; i < p.length; i++) {
			System.out.println(MatrixUtil.pretty(p[i]));
		}
		String title = sz + " Points Parzen Window Bin=" + bin + " with Normal Distribution( " + center + " , " + var
				+ " )";
		draw(samples, p, title);
	}

	public static void draw(double[] scatter, double[][] probs, String title) throws IOException {
		double[][] s = new double[scatter.length][];
		for (int i = 0; i < s.length; i++) {
			s[i] = new double[] { scatter[i], 0 };
		}
		JFreeChartUtils.drawLineAndScatter("tmp/parzen.png", title, s, probs);
	}

	public static double[] getRealDist(int sz, double center, double var) {
		double[] ret = new double[sz];
		double mean = center;
		double sd = var;
		NormalDistribution dist = new NormalDistribution(mean, sd);
		for (int i = 0; i < ret.length; i++) {
			ret[i] = dist.sample();
		}
		return ret;
	}

	public static double[][] ParzenKDE(int bin, double bound[], double[] samples) {
		double V = (bound[1] - bound[0]) / bin;
		double[][] ret = new double[bin][];
		int n = samples.length;
		for (int i = 0; i < bin; i++) {
			double[] bounds = new double[] { bound[0] + V * i, bound[0] + V * (i + 1) };
			ret[i] = estimateBin(samples, V, n, bounds);
		}
		return ret;
	}

	static double kernel(double x, double[] bounds) {
		return ((x >= bounds[0]) && (x <= bounds[1])) ? 1 : 0;
	}

	static double[] estimateBin(double[] samples, double V, int n, double[] bounds) {
		double k = 0d;
		for (double x : samples) {
			k += kernel(x, bounds);
		}
		return new double[] { (bounds[0] + bounds[1]) / 2, (k / n) / V };
	}

}
