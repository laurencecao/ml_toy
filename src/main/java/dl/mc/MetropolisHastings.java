package dl.mc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.util.FastMath;

import utils.DrawingUtils;

public class MetropolisHastings {

	final static double alpha = 2d;
	final static double beta = 4d;
	final static Function<Double, Double> compressedBETA = x -> {
		double ret = (alpha - 1) * FastMath.log(x) + (beta - 1) * FastMath.log(1 - x);
		return FastMath.exp(ret);
	};

	public static void main(String[] args) throws IOException {
		int TURN = 100000;
		List<Double> r = null;
		long ts = System.currentTimeMillis();
		r = MH(compressedBETA, TURN);
		long mhTs = System.currentTimeMillis() - ts;

		List<Double> r1 = null;
		r1 = r.subList(Double.valueOf(TURN * 0.999).intValue(), TURN - 1);
		System.out.println("Beta(" + alpha + "," + beta + "): \n" + r1);

		Double[] rrr = r.toArray(new Double[r.size()]);
		double[] mh = ArrayUtils.toPrimitive(rrr);
		ts = System.currentTimeMillis();
		double[] bt = realBeta(TURN, alpha, beta);
		long betaTs = System.currentTimeMillis() - ts;
		System.out.println("Sampling[" + TURN + "] MH: " + mhTs + "ms, BETA: " + betaTs + "ms");
		DrawingUtils.drawHistogram(Arrays.asList("MH", "BETA"), Arrays.asList(mh, bt), new double[] { 25d, 0d, 1d },
				"tmp/mh.png");
	}

	static List<Double> MH(Function<Double, Double> fnP, int turn) {
		// estimate Beta distribution
		// never mind the number of element in the transition matrix
		ThreadLocalRandom rng = ThreadLocalRandom.current();
		Double state = rng.nextDouble();
		ArrayList<Double> ret = new ArrayList<Double>();
		ret.add(state);
		Double next = null;
		int total = 0;
		for (int i = 0; i < turn; total++) {
			next = rng.nextDouble();
			Double pNow = fnP.apply(state);
			Double pNext = fnP.apply(next);
			// notice: Q_ij = Q_ji
			double accept = pNext / pNow;
			if (accept >= 1 || rng.nextDouble() < accept) {
				state = next;
				ret.add(state);
				i++;
			}
		}
		System.out.println("Total Sampling Turn: " + total);
		return ret;
	}

	static double[] realBeta(int turn, double a, double b) {
		BetaDistribution beta = new BetaDistribution(a, b);
		double[] ret = new double[turn];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = beta.sample();
		}
		return ret;
	}

}
