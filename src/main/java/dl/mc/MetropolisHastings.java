package dl.mc;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;

import org.apache.commons.math3.util.FastMath;

public class MetropolisHastings {

	final static double alpha = 10;
	final static double beta = 10;
	final static Function<Double, Double> BETA = x -> {
		return FastMath.pow(x, alpha) * FastMath.pow(1 - x, beta);
	};

	public static void main(String[] args) {
		int TURN = 10000;
		List<Double> r = null;
		r = MH(BETA, TURN);

		List<Double> r1 = null;
		r1 = r.subList(Double.valueOf(TURN * 0.999).intValue(), TURN - 1);
		System.out.println("Beta(" + alpha + "," + beta + "): \n" + r1);
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
			// notice: qij = qji
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

}
