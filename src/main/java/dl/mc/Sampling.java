package dl.mc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.function.Supplier;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.FastMath;

import utils.DrawingUtils;

public class Sampling {

	final static int SAMPLING_SIZE = 1000;
	final static double M = 8d;
	final static AtomicInteger name = new AtomicInteger();

	static Function<Double, Double> sin = x -> {
		return FastMath.sin(x);
	};

	static Function<Double, Double> fOut = x -> {
		// return -3 * FastMath.pow(x - 1, 2) + 9;
		return 1d;
	};

	static Function<Double, Double> fIn = x -> {
		return -3 * FastMath.pow(x - 1, 2) + 8;
	};

	static Function<Double, Double> flog = x -> {
		return FastMath.log(fIn.apply(x));
	};

	static Function<Double, Double> fdlog = x -> {
		// (-6x + 6) / (-3 * x^2 + 6x + 5)
		return (-6 * x + 6) / (-3 * FastMath.pow(x, 2) + 6 * x + 5);
	};

	public static void main(String[] args) throws IOException {
		double r = 0d;
		r = monteCarloIntegral(sin, 0, FastMath.PI);
		System.out.println("∫0->PI sin(x) dx ===> " + r);

		// range = (-0.5, 2)
		// extrema = 8 when x = 1
		r = rejectionSampling(fIn, fOut, -0.5d, 2d);
		System.out.println("rejection sampling: ∫-0.5->2 -3 * (x-1)^2 + 8 dx ===> " + r);
		r = monteCarloIntegral(fIn, -0.5d, 2d);
		System.out.println("monte carlo integral: ∫-0.5->2 -3 * (x-1)^2 + 8 dx ===> " + r);

		r = adaptiveRejectionSampling(fIn, flog, fdlog, 100, SAMPLING_SIZE, -0.5d, 2d);
		System.out.println("adaptive rejection sampling: ∫-0.5->2 -3 * (x-1)^2 + 8 dx ===> " + r);

		r = importanceSampling(fIn, -0.5d, 2d);
		System.out.println("importance sampling: ∫-0.5->2 -3 * (x-1)^2 + 8 dx  ===> " + r);
	}

	static double monteCarloIntegral(Function<Double, Double> fn, double a, double b) throws IOException {
		double[] val = new double[SAMPLING_SIZE];
		double[] idx = new double[SAMPLING_SIZE];
		SummaryStatistics ret = new SummaryStatistics();
		for (int i = 0; i < SAMPLING_SIZE; i++) {
			double x = ThreadLocalRandom.current().nextDouble(a, b);
			double y = fn.apply(x);
			ret.addValue(y);
			val[i] = ret.getMean() * (b - a);
			idx[i] = i;
		}
		DrawingUtils.drawSampling(val, idx, "tmp/mci" + name.incrementAndGet() + ".png",
				new String[] { "Sampling", "F(x)", "F(X)" });
		return ret.getMean() * (b - a);
	}

	static double rejectionSampling(Function<Double, Double> fnP, Function<Double, Double> fnOut, double a, double b)
			throws IOException {
		double[] val = new double[SAMPLING_SIZE];
		double[] idx = new double[SAMPLING_SIZE];
		SummaryStatistics ret = new SummaryStatistics();
		ThreadLocalRandom rnd = ThreadLocalRandom.current();
		int all = 0;
		for (int i = 0; i < SAMPLING_SIZE; all++) {
			double x = rnd.nextDouble(a, b);
			Double p = fnP.apply(x);
			double q = M * fnOut.apply(x);
			if (rnd.nextDouble() < (p / q)) {
				ret.addValue(p);
				val[i] = ret.getMean() * (b - a);
				idx[i] = i;
				i++;
			}
		}
		System.out.println("When M = " + M + ", Total Sampling Turn: " + all + "; Approved Turn: " + SAMPLING_SIZE);
		DrawingUtils.drawSampling(val, idx, "tmp/rs" + name.incrementAndGet() + ".png",
				new String[] { "Sampling", "F(x)", "F(X)" });
		return ret.getMean() * (b - a);
	}

	static double adaptiveRejectionSampling(Function<Double, Double> pdf, Function<Double, Double> logPdf,
			Function<Double, Double> derivativePdf, int num, int sample_size, double lower, double upper)
			throws IOException {
		double[] val = new double[SAMPLING_SIZE];
		double[] idx = new double[SAMPLING_SIZE];
		ARS sampler = new ARS(logPdf, derivativePdf);
		double step = (upper - lower) / (num - 1 + 2);
		for (double i = lower + step; i < upper; i += step) {
			sampler.xPoints.add(i);
			System.out.println("Addding X point: " + i + " ===> " + fdlog.apply(i));
		}

		SummaryStatistics sum = new SummaryStatistics();
		for (int i = 0; i < sample_size; i++) {
			double x = sampler.sample();
			Double y = pdf.apply(x);
			sum.addValue(y);
			val[i] = sum.getMean() * (upper - lower);
			idx[i] = i;
			if (i % 100 == 0) {
				System.out.println("sampling[" + i + "] = " + sum.getMean() * (upper - lower));
			}
		}
		DrawingUtils.drawSampling(val, idx, "tmp/ars" + name.incrementAndGet() + ".png",
				new String[] { "Sampling", "F(x)", "F(X)" });
		return sum.getMean() * (upper - lower);
	}

	static double importanceSampling(Function<Double, Double> fnY, double a, double b) throws IOException {
		double[] val = new double[SAMPLING_SIZE];
		double[] idx = new double[SAMPLING_SIZE];
		SummaryStatistics ret = new SummaryStatistics();
		ThreadLocalRandom rnd = ThreadLocalRandom.current();
		double lower = -2d;
		double upper = 8d;
		for (int i = 0; i < SAMPLING_SIZE; i++) {
			// original uniform distribution at [a, b]
			Double p = 1d / (b - a);
			// using another uniform distribution to estimate origin P(X)
			double x = rnd.nextDouble(lower, upper);
			// set to zero at (-∞, a) && (b, +∞)
			p = (x < a || x > b) ? 0 : p;
			double q = 1d / (upper - lower);
			double y = fnY.apply(x) * (p / q);
			ret.addValue(y);
			val[i] = ret.getMean() * (b - a);
			idx[i] = i;
		}
		DrawingUtils.drawSampling(val, idx, "tmp/is" + name.incrementAndGet() + ".png",
				new String[] { "Sampling", "F(x)", "F(X)" });
		return ret.getMean() * (b - a);
	}

}

/**
 * @see <a href="https://www.jstor.org/stable/2986186">Adaptive Rejection
 *      Sampling fromLog-concave Density Function</a>
 * @author caowenjiong
 *
 */
class ARS {

	List<Double> xPoints = new ArrayList<Double>();

	// input
	// log-concave function
	Function<Double, Double> h;

	// derivated log-concave function
	Function<Double, Double> dh;

	double lower;
	double upper;

	// inner below
	Function<Integer, Double> z = i -> {
		if (i == 0) {
			return lower;
		} else if (i == xPoints.size() - 1) {
			return upper;
		}

		Double xi = xPoints.get(i);
		Double xi1 = xPoints.get(i + 1);
		return xi + (h.apply(xi) - h.apply(xi1) + dh.apply(xi1) * (xi1 - xi)) / (dh.apply(xi1) - dh.apply(xi));
	};

	Function<Integer, Double> hU = i -> {
		Double xi = xPoints.get(i);
		return dh.apply(xi) * (z.apply(i) - xi) + h.apply(xi);
	};

	Function<Integer, Double> hL = i -> {
		double xz = z.apply(i);
		return h.apply(xz);
	};

	Supplier<Double> cNorm = () -> {
		Double ret = 0d;
		for (int i = 0; i < xPoints.size() - 1; i++) {
			Double xi1 = xPoints.get(i + 1);
			ret += (FastMath.exp(hU.apply(i + 1)) - FastMath.exp(hU.apply(i))) / dh.apply(xi1);
		}
		return ret;
	};

	Function<Integer, Double> sCum = i -> {
		Double ret = 0d;
		for (int idx = 0; idx < i; idx++) {
			Double xi1 = xPoints.get(idx + 1);
			ret += (FastMath.exp(hU.apply(idx + 1)) - FastMath.exp(hU.apply(idx))) / dh.apply(xi1);
		}
		return ret / cNorm.get();
	};

	public ARS(Function<Double, Double> h, Function<Double, Double> dh) {
		this.h = h;
		this.dh = dh;
	}

	public double sample() {
		ThreadLocalRandom rnd = ThreadLocalRandom.current();
		while (true) {
			double u0 = rnd.nextDouble();
			int xIdx = sIndex(u0);
			double x = sValue(xIdx, u0);
			double u1 = rnd.nextDouble();

			if (u1 < FastMath.exp(hL.apply(xIdx) - hU.apply(xIdx))) {
				// accept
				return x;
			} else {
				// perform rejection testing
				if (u1 < FastMath.exp(h.apply(x) - hU.apply(xIdx))) {
					// accept
					return x;
				}

				// updating the support points
				updateHull(x);
			}
		}
	}

	public int sIndex(double u) {
		int i = 0;
		for (i = 0; i < xPoints.size() - 1; i++) {
			double t = sCum.apply(i);
			if (t < u) {
				if (i == xPoints.size() - 1) {
					break;
				}
				t = sCum.apply(i + 1);
				if (t > u) {
					break;
				}
			}
		}
		return i;
	}

	public double sValue(int i, double u) {
		double xi1 = xPoints.get(xPoints.size() - 1);
		if (i + 1 < xPoints.size()) {
			xi1 = xPoints.get(i + 1);
		}
		double ret = z.apply(i) + (1 / dh.apply(xi1))
				* FastMath.log(1 + dh.apply(xi1) * cNorm.get() * (u - sCum.apply(i)) / FastMath.exp(hU.apply(i)));
		return ret;
	}

	public void updateHull(double x) {
		xPoints.add(x);
		Collections.sort(xPoints);
	}

}
