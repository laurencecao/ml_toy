package dl.hmm;

import java.util.Arrays;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class FourCoinTest {

	final static double precision = 0.0000001d;
	ContextHMM ctx;
	RealVector[] data;

	@Before
	public void setUp() throws Exception {
		RealVector[] d = new RealVector[1];
		// d[0] = MatrixUtils.createRealVector(new double[] { 1, 0, 1 });
		d[0] = MatrixUtils.createRealVector(new double[] { 0, 1, 1, 0 });
		data = d;
		ctx = FourCoin.initContext(d, 2, 2);
		ctx.initialStates.setEntry(0, 0.85d);
		ctx.initialStates.setEntry(1, 0.15d);

		ctx.transitionProbs.setEntry(0, 0, 0.3d);
		ctx.transitionProbs.setEntry(0, 1, 0.7d);
		ctx.transitionProbs.setEntry(1, 0, 0.1d);
		ctx.transitionProbs.setEntry(1, 1, 0.9d);

		ctx.emissionProbs.setEntry(0, 0, 0.4d);
		ctx.emissionProbs.setEntry(0, 1, 0.6d);
		ctx.emissionProbs.setEntry(1, 0, 0.5d);
		ctx.emissionProbs.setEntry(1, 1, 0.5d);
	}

	@After
	public void tearDown() throws Exception {
	}

	// @Test
	public void testAlpha() {
		ContextHMM c1 = ctx.copy();
		FourCoin.alpha(data[0], c1);
		System.out.println(c1.alpha);

		ContextHMM c2 = ctx.copy();
		FourCoin.alphaV2(data[0], c2);
		System.out.println(c2.alpha);

		for (int i = 0; i < c1.alpha.getRowDimension(); i++) {
			for (int j = 0; j < c1.alpha.getColumnDimension(); j++) {
				double d1 = c1.alpha.getEntry(i, j);
				double d2 = FastMath.exp(c2.alpha.getEntry(i, j));
				Assert.assertTrue(FastMath.abs(d1 - d2) < precision);
			}
		}
	}

	// @Test
	public void testBeta() {
		ContextHMM c1 = ctx.copy();
		FourCoin.beta(data[0], c1);
		System.out.println(c1.beta);

		ContextHMM c2 = ctx.copy();
		FourCoin.betaV2(data[0], c2);
		System.out.println(c2.beta);

		for (int i = 0; i < c1.beta.getRowDimension(); i++) {
			for (int j = 0; j < c1.beta.getColumnDimension(); j++) {
				double d1 = c1.beta.getEntry(i, j);
				double d2 = FastMath.exp(c2.beta.getEntry(i, j));
				Assert.assertTrue(FastMath.abs(d1 - d2) < precision);
			}
		}
	}

	// @Test
	public void testGamma() {
		ContextHMM c1 = ctx.copy();
		FourCoin.alpha(data[0], c1);
		FourCoin.beta(data[0], c1);
		FourCoin.gamma(data[0], c1);
		System.out.println(c1.gamma);

		ContextHMM c2 = ctx.copy();
		FourCoin.alphaV2(data[0], c2);
		FourCoin.betaV2(data[0], c2);
		FourCoin.gammaV2(data[0], c2);
		System.out.println(c2.gamma);

		for (int i = 0; i < c1.gamma.getRowDimension(); i++) {
			for (int j = 0; j < c1.gamma.getColumnDimension(); j++) {
				double d1 = c1.gamma.getEntry(i, j);
				double d2 = FastMath.exp(c2.gamma.getEntry(i, j));
				Assert.assertTrue(FastMath.abs(d1 - d2) < precision);
			}
		}
	}

	// @Test
	public void testKsi() {
		ContextHMM c1 = ctx.copy();
		FourCoin.alpha(data[0], c1);
		FourCoin.beta(data[0], c1);
		FourCoin.gamma(data[0], c1);
		FourCoin.ksi(data[0], c1);
		for (int t = 0; t < c1.ksi.length; t++) {
			System.out.println(c1.ksi[t]);
		}

		System.out.println("----------------------------------------");

		ContextHMM c2 = ctx.copy();
		FourCoin.alphaV2(data[0], c2);
		FourCoin.betaV2(data[0], c2);
		FourCoin.gammaV2(data[0], c2);
		FourCoin.ksiV2(data[0], c2);
		for (int t = 0; t < c2.ksi.length; t++) {
			System.out.println(c2.ksi[t]);
		}

		for (int t = 0; t < c1.ksi.length; t++) {
			for (int i = 0; i < c1.ksi[t].getRowDimension(); i++) {
				for (int j = 0; j < c1.ksi[t].getColumnDimension(); j++) {
					double d1 = c1.ksi[t].getEntry(i, j);
					double d2 = FastMath.exp(c2.ksi[t].getEntry(i, j));
					Assert.assertTrue(FastMath.abs(d1 - d2) < precision);
				}
			}
		}
	}

	// @Test
	public void testSumAtLogSpace() {
		double r = ContextHMM.sumAtLogSpace(new double[] { 2, 4, 0 });
		System.out.println(r);
		r = FastMath.log(FastMath.exp(2) + FastMath.exp(4) + FastMath.exp(0));
		System.out.println(r);
		r = FastMath.log(FastMath.exp(r) - 1);
		System.out.println(r);
		r = FastMath.log(FastMath.exp(2) + FastMath.exp(4));
		System.out.println(r);
	}

	// @Test
	public void testPI() {
		ContextHMM c1 = ctx.copy();
		FourCoin.alpha(data[0], c1);
		FourCoin.beta(data[0], c1);
		FourCoin.gamma(data[0], c1);
		FourCoin.ksi(data[0], c1);
		RealVector pi1 = FourCoin.estimatePI(new ContextHMM[] { c1 });
		System.out.println(pi1);

		ContextHMM c2 = ctx.copy();
		FourCoin.alphaV2(data[0], c2);
		FourCoin.betaV2(data[0], c2);
		FourCoin.gammaV2(data[0], c2);
		FourCoin.ksiV2(data[0], c2);
		RealVector pi2 = FourCoin.estimatePIV2(new ContextHMM[] { c2 });
		System.out.println(pi2);

		for (int i = 0; i < pi1.toArray().length; i++) {
			double d1 = pi1.toArray()[i];
			double d2 = pi2.toArray()[i];
			Assert.assertTrue(FastMath.abs(d1 - d2) < precision);
		}
	}

	// @Test
	public void testTransition() {
		ContextHMM c1 = ctx.copy();
		FourCoin.alpha(data[0], c1);
		FourCoin.beta(data[0], c1);
		FourCoin.gamma(data[0], c1);
		FourCoin.ksi(data[0], c1);
		RealMatrix t1 = FourCoin.estimateT(new ContextHMM[] { c1 });
		System.out.println(t1);

		ContextHMM c2 = ctx.copy();
		FourCoin.alphaV2(data[0], c2);
		FourCoin.betaV2(data[0], c2);
		FourCoin.gammaV2(data[0], c2);
		FourCoin.ksiV2(data[0], c2);
		RealMatrix t2 = FourCoin.estimateTV2(new ContextHMM[] { c2 });
		System.out.println(t2);

		for (int i = 0; i < t1.getRowDimension(); i++) {
			for (int j = 0; j < t1.getColumnDimension(); j++) {
				double d1 = t1.getEntry(i, j);
				double d2 = t2.getEntry(i, j);
				Assert.assertTrue("d1: " + d1 + ", d2: " + d2, FastMath.abs(d1 - d2) < precision);
			}
		}
	}

	// @Test
	public void testEmission() {
		ContextHMM c1 = ctx.copy();
		FourCoin.alpha(data[0], c1);
		FourCoin.beta(data[0], c1);
		FourCoin.gamma(data[0], c1);
		FourCoin.ksi(data[0], c1);
		RealMatrix t1 = FourCoin.estimateQ(data, new ContextHMM[] { c1 });
		System.out.println(t1);

		ContextHMM c2 = ctx.copy();
		FourCoin.alphaV2(data[0], c2);
		FourCoin.betaV2(data[0], c2);
		FourCoin.gammaV2(data[0], c2);
		FourCoin.ksiV2(data[0], c2);
		RealMatrix t2 = FourCoin.estimateQV2(data, new ContextHMM[] { c2 });
		System.out.println(t2);

		for (int i = 0; i < t1.getRowDimension(); i++) {
			for (int j = 0; j < t1.getColumnDimension(); j++) {
				double d1 = t1.getEntry(i, j);
				double d2 = t2.getEntry(i, j);
				Assert.assertTrue("d1: " + d1 + ", d2: " + d2, FastMath.abs(d1 - d2) < precision);
			}
		}
	}

	// @Test
	public void testEvaluation() {
		ContextHMM c1 = ctx.copy();
		double v1 = FourCoin.evaluation(data[0], c1);
		System.out.println(v1);

		ContextHMM c2 = ctx.copy();
		double v2 = FourCoin.evaluationV2(data[0], c2);
		System.out.println(v2);

		Assert.assertTrue("v1: " + v1 + ", v2: " + v2, FastMath.abs(v1 - v2) < precision);
	}

	// @Test
	public void testDecoding() {
		ContextHMM c1 = ctx.copy();
		int[] v1 = FourCoin.decoding(data[0], c1);
		System.out.println(Arrays.toString(v1));

		ContextHMM c2 = ctx.copy();
		int[] v2 = FourCoin.decodingV2(data[0], c2);
		System.out.println(Arrays.toString(v2));

		Assert.assertTrue("v1: " + Arrays.toString(v1) + ", v2: " + Arrays.toString(v2),
				StringUtils.equalsAnyIgnoreCase(Arrays.toString(v1), Arrays.toString(v2)));
	}

	// @Test
	public void testRun() {
		ContextHMM c = ctx.copy();

		ContextHMM model = FourCoin.learning(data, c, 1);
		int[] o = null;
		o = FourCoin.decodingV2(data[0], model);
		System.out.println(Arrays.toString(o));
	}

	// @Test
	public void testRun1() {
		ContextHMM model = ctx.copy();

		for (int i = 0; i < 10; i++) {
			FourCoin.eStepV2(data[0], model);
			model = FourCoin.mStepV2(data, new ContextHMM[] { model });
			int[] o = null;
			o = FourCoin.decodingV2(data[0], model);
			System.out.println(Arrays.toString(o));
		}

	}

	// @Test
	public void testRun2() {
		ContextHMM model = ctx.copy();

		for (int i = 0; i < 10; i++) {
			FourCoin.eStep(data[0], model);
			model = FourCoin.mStep(data, new ContextHMM[] { model });
			int[] o = null;
			o = FourCoin.decodingV2(data[0], model);
			System.out.println(model.transitionProbs);
			System.out.println(Arrays.toString(o));
		}

	}

	@Test
	public void testViterbi() {
		ContextHMM model = ctx.copy();

		for (int i = 0; i < 10; i++) {
			model = FourCoinViterbi.mStep(data, model);
			int[] o = null;
			o = FourCoin.decodingV2(data[0], model);
			System.out.println();
			System.out.println(model.initialStates);
			System.out.println(model.transitionProbs);
			System.out.println(model.emissionProbs);
			System.out.println(Arrays.toString(o));
			System.out.println();
		}

	}

}
