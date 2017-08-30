package dl.em;

import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.FastMath;

import dataset.NNDataset;

public class EM3Coins {

	final static double ERROR = 0.001d;
	final static int debug = 100;

	public static void main(String[] args) {
		RealVector[] data = NNDataset.getData(NNDataset.THREECOINS);
		ThreeCoins ctx = training(data);
		String s = "CoinA=" + ctx.pA + ", CoinB=" + ctx.pB + ", CoinC=" + ctx.pC;
		System.out.println(s);
	}

	static ThreeCoins training(RealVector[] data) {

		// initialization coins
		ThreeCoins ctx = new ThreeCoins();
		ctx.sampleSize = data.length;
		ctx.batchSize = data[0].getDimension();
		ctx.pA = ctx.rng.nextDouble(0.1d);
		ctx.pB = ctx.rng.nextDouble(0.1d);
		ctx.pC = ctx.rng.nextDouble(0.1d);
		ctx.iden = MatrixUtils.createRealVector(new double[ctx.batchSize]);
		ctx.iden.set(1);
		ctx.dC = new double[ctx.sampleSize];
		ctx.dX = new double[ctx.sampleSize];

		int epoch = 0;
		boolean terminated = false;
		while (!terminated) {
			epoch++;
			/**
			 * <pre>
			 * Q_t(θ) = Σz p(z|X;θ(t')) * log(P(z,X;θ)
			 * 	 θ(t) <- maximize θ(t'')
			 * argmax(θ) w.r.t Σz p(z|X;θ(t')) * log(P(z,X;θ)
			 * </pre>
			 */

			// E(Z|X,θ)
			stepExpectation(data, ctx);

			// E(θ|X,Z')
			stepMaximization(data, ctx);

			RealVector last = MatrixUtils.createRealVector(new double[] { ctx._pA, ctx._pB, ctx._pC });
			RealVector current = MatrixUtils.createRealVector(new double[] { ctx.pA, ctx.pB, ctx.pC });

			double err = current.subtract(last).getL1Norm();
			if (err < ERROR) {
				terminated = true;
			}

			if (epoch % debug == 0 || terminated) {
				String s = "CoinA=" + ctx.pA + ", CoinB=" + ctx.pB + ", CoinC=" + ctx.pC;
				System.out.println("epoch[" + epoch + "] --> " + s);
			}
		}
		return ctx;
	}

	static void stepMaximization(RealVector[] data, ThreeCoins ctx) {
		SummaryStatistics a_element = new SummaryStatistics();
		SummaryStatistics a_denominatore = new SummaryStatistics();
		SummaryStatistics b_element = new SummaryStatistics();
		SummaryStatistics b_denominatore = new SummaryStatistics();

		// p = Σ(u * X) / Σ(u * N)
		for (int i = 0; i < data.length; i++) {
			a_element.addValue(ctx.dC[i] * ctx.dX[i]);
			a_denominatore.addValue(ctx.dC[i] * ctx.batchSize);
			b_element.addValue((1 - ctx.dC[i]) * ctx.dX[i]);
			b_denominatore.addValue((1 - ctx.dC[i]) * ctx.batchSize);
		}
		ctx._pA = ctx.pA;
		ctx._pB = ctx.pB;
		ctx.pA = a_element.getSum() / a_denominatore.getSum();
		ctx.pB = b_element.getSum() / b_denominatore.getSum();
	}

	static void stepExpectation(RealVector[] data, ThreeCoins ctx) {
		// 1. estimate every coin_c's expectation => E_C{i}
		// 2. estimate expectation of E_C
		SummaryStatistics sum = new SummaryStatistics();
		for (int i = 0; i < data.length; i++) {
			RealVector d = data[i];
			double count_1 = ctx.iden.dotProduct(d);
			double _1 = ctx.pC * FastMath.pow(ctx.pA, count_1) * FastMath.pow(1 - ctx.pA, ctx.batchSize - count_1);
			double _2 = (1 - ctx.pC) * FastMath.pow(ctx.pB, count_1)
					* FastMath.pow(1 - ctx.pB, ctx.batchSize - count_1);
			ctx.dC[i] = _1 / (_1 + _2);
			ctx.dX[i] = count_1;
			sum.addValue(ctx.dC[i]);
		}
		ctx._pC = ctx.pC;
		ctx.pC = sum.getMean();
	}

}

class ThreeCoins {

	ThreadLocalRandom rng = ThreadLocalRandom.current();

	RealVector iden;
	int sampleSize;
	int batchSize;

	double pA;
	double pB;
	double pC;

	double _pA;
	double _pB;
	double _pC;

	double[] dC;
	double[] dX;

}