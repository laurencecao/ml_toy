package dl.mc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.util.FastMath;

import utils.DrawingUtils;

public class GibbsSampling {

	final static double uA = -2d;
	final static double uB = 4d;
	final static double tA = 1.1d;
	final static double tB = 2.3d;
	final static double cor = 0.9d;

	final static int SAMPLE_SIZE = 100000;
	final static int BURN_IN = Double.valueOf(0.1d * SAMPLE_SIZE).intValue();

	public static void main(String[] args) {
		List<double[]> r1 = null;
		r1 = gibbs();
		r1 = r1.subList(BURN_IN, SAMPLE_SIZE);
		List<double[]> r2 = null;
		r2 = multivariate();

		double[][] data1 = r1.toArray(new double[r1.size()][]);
		double[][] data2 = r2.toArray(new double[r2.size()][]);
		DrawingUtils.draw3DHistogram(new String[] { "gibbs sampling", "multibivariate normal" }, 100, data1, data2,
				"tmp/gibbs.png");
	}

	static List<double[]> gibbs() {
		ThreadLocalRandom rng = ThreadLocalRandom.current();
		double[] state = new double[] { rng.nextDouble(), rng.nextDouble() };
		ArrayList<double[]> ret = new ArrayList<double[]>();
		ret.add(state);
		for (int i = 0; i < SAMPLE_SIZE; i++) {
			state[1] = getCondition(state[0], uA, tA, uB, tB, cor);
			state[0] = getCondition(state[1], uB, tB, uA, tA, cor);
			ret.add(Arrays.copyOf(state, state.length));
		}
		return ret;
	}

	// P(B|A)
	static double getCondition(double x, double uA, double tA, double uB, double tB, double rou) {
		double u = uB + rou * (tB / tA) * (x - uA);
		double t = FastMath.pow(tB, 2) * (1 - FastMath.pow(rou, 2));
		NormalDistribution nd = new NormalDistribution(u, t);
		return nd.sample();
	}

	static List<double[]> multivariate() {
		MultivariateNormalDistribution d = new MultivariateNormalDistribution(new double[] { uA, uB },
				new double[][] { { FastMath.pow(tA, 2), cor * tA * tB }, { cor * tA * tB, FastMath.pow(tB, 2) } });
		List<double[]> ret = new ArrayList<double[]>();
		for (int i = 0; i < SAMPLE_SIZE - BURN_IN; i++) {
			ret.add(d.sample());
		}
		return ret;
	}

}
