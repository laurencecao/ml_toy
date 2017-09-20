package dl.mc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
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

	final static double VERY_SMALL = 0.00001d;
	final static String[] dataset = new String[] { "GTAAACAATATTTATAGC", "AAAATTTACCTCGCAAGG", "CCGTACTGTCAAGCGTGG",
			"TGAGTAATCGACGTCCCA", "TACTTCACACCCTGTCAA" };

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

		motifFinding(dataset, 5);
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

	/**
	 * @see Bioinformatics
	 * @param sequence
	 * @param motifSize
	 */
	static void motifFinding(String[] sequence, int motifSize) {
		MotifProfiler profiler = new MotifProfiler(motifSize);
		profiler.align(sequence);

		double last = 1000d;
		double epsilon = 0.00001d;
		boolean convegenced = false;
		while (!convegenced) {
			double likelihood = profiler.sample(sequence);
			System.out.println("now likelihood: " + likelihood);
			if (FastMath.abs(last - likelihood) < epsilon) {
				convegenced = true;
				break;
			}
			last = likelihood;
		}

		String s;
		for (int i = 0; i < sequence.length; i++) {
			int idx = profiler.sPosition[i];
			s = sequence[i].substring(idx, idx + profiler.motifSize);
			System.out.println(sequence[i] + " =====> " + s);
		}
		String[] ss = new String[profiler.motifSize];
		for (int i = 0; i < profiler.motifSize; i++) {
			RealVector v = profiler.model.getColumnVector(i);
			double p = -1d;
			for (int j = 0; j < v.getDimension(); j++) {
				if (v.getEntry(j) > p) {
					ss[i] = MotifProfiler.SYMBOLS.get(j);
					p = v.getEntry(j);
				}
			}
		}
		s = Arrays.toString(ss).replace(",", "").replace(" ", "").replace("[", "").replace("]", "");
		System.out.println("MOTIF[" + profiler.motifSize + "] =====> " + s);
		for (int i = 0; i < MotifProfiler.SYMBOLS.size(); i++) {
			String sym = MotifProfiler.SYMBOLS.get(i);
			System.out.println(sym + " ==> " + i);
		}
		s = Arrays.toString(profiler.background.toArray());
		System.out.println("background: " + s);
		double[][] data = profiler.model.getData();
		System.out.println("model: ");
		for (int i = 0; i < data.length; i++) {
			s = Arrays.toString(data[i]);
			System.out.println(s);
		}
	}

}

class MotifProfiler {

	final static List<String> SYMBOLS = Arrays.asList("ACGT".split(""));

	int motifSize;
	RealVector background;
	RealMatrix model;

	int[] sPosition;

	MotifProfiler(int motifSize) {
		this.motifSize = motifSize;
		this.background = MatrixUtils.createRealVector(new double[SYMBOLS.size()]);
		this.model = MatrixUtils.createRealMatrix(SYMBOLS.size(), motifSize);
	}

	public void align(String[] sequence) {
		this.sPosition = new int[sequence.length];
		for (int i = 0; i < this.sPosition.length; i++) {
			sPosition[i] = ThreadLocalRandom.current().nextInt(sequence[i].length() - motifSize + 1);
		}
	}

	public void computeBM(String seq, int seqIdx) {
		String[] s = seq.split("");
		for (int i = 0; i < s.length; i++) {
			int idx = SYMBOLS.indexOf(s[i]);
			double v = background.getEntry(idx) + 1;
			background.setEntry(idx, v);
		}
		for (int j = 0; j < motifSize; j++) {
			int idx = SYMBOLS.indexOf(s[sPosition[seqIdx] + j]);
			double v = model.getEntry(idx, j) + 1;
			model.setEntry(idx, j, v);
		}
	}

	public double sample(String[] sequence) {
		int h = ThreadLocalRandom.current().nextInt(sequence.length);
		background.set(0);
		model = model.scalarMultiply(0d);
		for (int i = 0; i < sequence.length; i++) {
			if (i == h) {
				continue;
			}
			computeBM(sequence[i], i);
		}
		RealVector idenV = MatrixUtils.createRealVector(new double[background.getDimension()]);
		idenV.set(1d);
		double deno = idenV.dotProduct(background);
		background.mapDivideToSelf(deno);
		model.scalarMultiply(FastMath.pow(sequence.length - 1, -1));

		String[] seq = sequence[h].split("");
		double prob = -1d;
		int s = 0;
		for (int i = 0; i < seq.length - motifSize + 1; i++) {
			Double p = 0d; // log space
			for (int j = 0; j < motifSize; j++) {
				int idx = SYMBOLS.indexOf(seq[i + j]);
				double q = model.getEntry(idx, j);
				if (p == null || q == 0) {
					p = null;
				} else {
					p += FastMath.log(q);
				}
			}
			p = p == null ? GibbsSampling.VERY_SMALL : p;
			if (p > prob) {
				s = i;
				prob = p;
			}
		}
		sPosition[h] = s;

		background.set(0);
		model.scalarMultiply(0d);
		for (int i = 0; i < sequence.length; i++) {
			computeBM(sequence[i], i);
		}
		deno = idenV.dotProduct(background);
		background.mapDivideToSelf(deno);
		RealMatrix idenM = MatrixUtils.createRealMatrix(1, SYMBOLS.size());
		idenM = idenM.scalarAdd(1d);
		idenM = idenM.multiply(model);
		for (int i = 0; i < SYMBOLS.size(); i++) {
			RealVector v = idenM.getRowVector(0);
			RealVector motif = model.getRowVector(i);
			motif = motif.ebeDivide(v);
			model.setRowVector(i, motif);
		}
		// model = model.scalarMultiply(FastMath.pow(sequence.length, -1));

		// notice: return at log space
		return likelihood(sequence);
	}

	public double likelihood(String[] sequence) {
		double ret = 0d;
		for (int i = 0; i < sequence.length; i++) {
			String[] seq = sequence[i].split("");
			double p = 0d; // log space
			for (int j = 0; j < sPosition[i]; j++) {
				int idx = SYMBOLS.indexOf(seq[j]);
				double _p = background.getEntry(idx);
				_p = _p == 0 ? GibbsSampling.VERY_SMALL : _p;
				p += FastMath.log(_p);
			}
			for (int j = 0; j < motifSize; j++) {
				int idx = SYMBOLS.indexOf(seq[sPosition[i] + j]);
				double _p = model.getEntry(idx, j);
				_p = _p == 0 ? GibbsSampling.VERY_SMALL : _p;
				p += FastMath.log(_p);
			}
			for (int j = sPosition[i] + motifSize; j < seq.length; j++) {
				int idx = SYMBOLS.indexOf(seq[j]);
				double _p = background.getEntry(idx);
				_p = _p == 0 ? GibbsSampling.VERY_SMALL : _p;
				p += FastMath.log(_p);
			}
			ret += p;
		}
		return ret;
	}

	public MotifProfiler copy() {
		MotifProfiler ret = new MotifProfiler(this.motifSize);
		ret.background = background.copy();
		ret.model = model.copy();
		return ret;
	}

}