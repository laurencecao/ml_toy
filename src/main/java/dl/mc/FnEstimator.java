package dl.mc;

import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;

import org.apache.commons.math3.util.FastMath;

public class FnEstimator {

	final static long SAMPLING_SIZE = 1000000;

	public static void main(String[] args) {
		Function<Double, Double> integral_sin = x -> {
			// notice: using derivation
			// [-cos(x)]' = sin(x)
			return -FastMath.cos(x);
		};
		double r = 0d;
		r = classicalIntegral(integral_sin, FastMath.PI, 0);
		System.out.println(r);

		Function<Double, Double> sin = x -> {
			return FastMath.sin(x);
		};
		r = monteCarloIntegral(x -> {
			return Math.exp(x);
		}, 1, 0);
		System.out.println(r);
		r = monteCarloIntegral(sin, FastMath.PI, 0);
		System.out.println(r);
	}

	public static double monteCarloIntegral(Function<Double, Double> fn, double upper, double lower) {
		// ∫a->b f(x) dx
		// = (b-a) ∫a->b f(x) / (b-a) dx
		// = (b-a) ∫a->b f(x) * p(x) dx
		// = (b-a)/N * Σf(X_i)

		double ret = 0;
		// avg1 = (avg0 * size + new_x) / (size + 1)
		// = avg0 * size / (size + 1) + new_x / (size + 1)
		// caution: because very large error in first part
		// using:
		// = (avg0 * size + avg0 + new_x - avg0) / (size + 1)
		// = avg0 + (new_x - avg0) / (size + 1)
		/**
		 * @See Moving Average
		 */
		for (long i = 0; i < SAMPLING_SIZE; i++) {
			double x = ThreadLocalRandom.current().nextDouble(lower, upper);
			ret = ret + (fn.apply(x) - ret) / (i + 1);
		}
		return (upper - lower) * ret;
	}

	public static double classicalIntegral(Function<Double, Double> origin_fn, double upper, double lower) {
		double a = origin_fn.apply(upper);
		double b = origin_fn.apply(lower);
		return a - b;
	}

}
