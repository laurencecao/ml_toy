package dl.hmm;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;

import dataset.NNDataset;

public class FourCoinViterbi {

	final static int TURN = 5000;
	final static double EPSILON = 0.001d;
	final static int DEBUG = 10;
	final static double VERYSMALL = 0.00000001d;

	/**
	 * second order Hidden Markov Model
	 * 
	 * 
	 * sigh! very sensitive to initial parameters
	 * 
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		RealVector[] seq = NNDataset.getData(NNDataset.FOURCOINS);
		// foobar1();
		// foobar2();
		// boolean b = true;
		// if (b) {
		// System.exit(0);
		// }

		/**
		 * state = {A, B}; symbol = {T, H}
		 */
		ContextHMM origin = initContext(seq, 2, 2);
		System.out.println("Before training ............ ");
		print(origin);
		ContextHMM model = learning(seq, origin, DEBUG);
		System.out.println("After training .......... ");
		print(model);

		// "HHTHHHHHTH" -> 2
		RealVector s1 = MatrixUtils.createRealVector(new double[] { 1, 1, 0, 1, 1, 1, 1, 1, 0, 1 });
		// "TTTTTHTHHH" -> 1
		RealVector s2 = MatrixUtils.createRealVector(new double[] { 0, 0, 0, 0, 0, 1, 0, 1, 1, 1 });

		System.out.println("HHTHHHHHTH => " + evaluation(s1, model));
		System.out.println("TTTTTHTHHH => " + evaluation(s2, model));

		int[] symbol = null;
		symbol = decodingV2(s1, model);
		System.out.println(Arrays.toString(symbol));
		symbol = decodingV2(s2, model);
		System.out.println(Arrays.toString(symbol));
	}

	static void print(ContextHMM model) {
		String s = "Initial CoinA => " + model.initialStates.getEntry(0);
		System.out.println(s);
		s = "Initial CoinB => " + model.initialStates.getEntry(1);
		System.out.println(s);

		s = "Transition Table: ";
		System.out.println(s);

		s = "CoinA => CoinA: " + model.transitionProbs.getEntry(0, 0);
		System.out.println(s);
		s = "CoinA => CoinB: " + model.transitionProbs.getEntry(0, 1);
		System.out.println(s);
		s = "CoinB => CoinA: " + model.transitionProbs.getEntry(1, 0);
		System.out.println(s);
		s = "CoinB => CoinB: " + model.transitionProbs.getEntry(1, 1);
		System.out.println(s);

		s = "Emission Table: ";
		System.out.println(s);
		s = "CoinA => HEAD: " + model.emissionProbs.getEntry(0, 1);
		System.out.println(s);
		s = "CoinA => TAIL: " + model.emissionProbs.getEntry(0, 0);
		System.out.println(s);
		s = "CoinB => HEAD: " + model.emissionProbs.getEntry(1, 1);
		System.out.println(s);
		s = "CoinB => TAIL: " + model.emissionProbs.getEntry(1, 0);
		System.out.println(s);
	}

	static void foobar1() {
		RealVector[] d = new RealVector[1];
		d[0] = MatrixUtils.createRealVector(new double[] { 0, 1, 1, 0 });
		ContextHMM origin = initContext(d, 2, 2);
		origin.initialStates.setEntry(0, 0.85d);
		origin.initialStates.setEntry(1, 0.15d);
		System.out.println("Before training ............ ");
		print(origin);
		ContextHMM model = learning(d, origin, DEBUG);
		System.out.println("After training .......... ");
		print(model);

		RealVector s2 = d[0];
		System.out.println("0110 => " + evaluation(s2, model));

		int[] symbol = null;
		symbol = decodingV2(s2, model);
		System.out.println(Arrays.toString(symbol));

	}

	static void foobar2() {
		RealVector[] d = new RealVector[1];
		// d[0] = MatrixUtils.createRealVector(new double[] { 1, 0, 1 });
		d[0] = MatrixUtils.createRealVector(new double[] { 0, 1, 1, 0 });
		ContextHMM ctx = initContext(d, 2, 2);
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

		int[] demo = decodingV2(d[0], ctx);
		System.out.println(Arrays.toString(demo));

		for (int i = 0; i < 100; i++) {
			ContextHMM c = mStep(d, ctx);
			demo = decodingV2(d[0], c);
			System.out.println("decoding at [" + i + "] --> " + Arrays.toString(demo));
			ctx = c;

			if (i % 10 == 0) {
				print(c);
			}
		}
		print(ctx);

		System.out.println(Arrays.toString(d[0].toArray()) + " => " + evaluation(d[0], ctx));

		int[] symbol = null;
		symbol = decodingV2(d[0], ctx);
		System.out.println(Arrays.toString(symbol));
	}

	static ContextHMM initContext(RealVector[] sequence, int state_size, int symbol_size) {
		ContextHMM ret = new ContextHMM(sequence[0].getDimension(), state_size, symbol_size);
		RealMatrix init = initMatrix(ret.state_size, 1, true);
		ret.initialStates = normalizeByColumn(init).getColumnVector(0);
		// ret.initialStates.set(1d / state_size);
		ret.transitionProbs = initMatrix(ret.state_size, ret.state_size, true);
		ret.transitionProbs = normalizeByColumn(ret.transitionProbs.transpose()).transpose();
		ret.emissionProbs = initMatrix(ret.state_size, ret.symbol_size, true);
		ret.emissionProbs = normalizeByColumn(ret.emissionProbs.transpose()).transpose();
		ret.alpha = initMatrix(ret.state_size, ret.batch_size, false);
		ret.beta = initMatrix(ret.state_size, ret.batch_size, false);
		ret.gamma = initMatrix(ret.state_size, ret.batch_size, false);
		ret.ksi = new RealMatrix[ret.batch_size - 1];
		for (int i = 0; i < ret.ksi.length; i++) {
			ret.ksi[i] = initMatrix(ret.state_size, ret.state_size, false);
		}
		return ret;
	}

	static RealMatrixChangingVisitor rnd = new RealMatrixChangingVisitor() {

		@Override
		public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
		}

		@Override
		public double visit(int row, int column, double value) {
			double ret = ThreadLocalRandom.current().nextDouble();
			while (ret < 0.1d) {
				ret = ThreadLocalRandom.current().nextDouble();
			}
			return ret;
		}

		@Override
		public double end() {
			return 0;
		}

	};

	static RealMatrix initMatrix(int row, int col, boolean initial) {
		RealMatrix ret = MatrixUtils.createRealMatrix(row, col);
		if (initial) {
			ret.walkInOptimizedOrder(rnd);
		}
		return ret;
	}

	static RealMatrix normalizeByColumn(RealMatrix m) {
		RealMatrix ret = m.copy();
		RealVector iden = MatrixUtils.createRealVector(new double[ret.getRowDimension()]);
		iden.set(1);
		for (int i = 0; i < ret.getColumnDimension(); i++) {
			RealVector v = ret.getColumnVector(i);
			double deno = iden.dotProduct(v);
			v.mapDivideToSelf(deno);
			ret.setColumnVector(i, v);
		}
		return ret;
	}

	static double checkConvergence(ContextHMM c1, ContextHMM c2) {
		double n0 = c1.initialStates.subtract(c2.initialStates).getNorm();
		double n1 = c1.transitionProbs.subtract(c2.transitionProbs).getNorm();
		double n2 = c1.emissionProbs.subtract(c2.emissionProbs).getNorm();
		return (n0 + n1 + n2) / 3;
	}

	/**
	 * Viterbi Training Algorithm
	 * 
	 * @param sequence
	 */
	static ContextHMM learning(RealVector[] sequence, ContextHMM origin, int debug) {
		ContextHMM ret = origin.copy();
		int t = 0;
		for (t = 0; t < TURN; t++) {

			ContextHMM ctx = mStep(sequence, ret);
			double eps = checkConvergence(ret, ctx);

			ret = ctx;

			if (t % debug == 0) {
				System.out.println("Debug information output: " + t);
				print(ret);
			}

			if (eps < EPSILON) {
				break;
			}

		}
		System.out.println("Total Learning Turn: " + t);
		return ret.copy();
	}

	static ContextHMM mStep(RealVector[] seq, ContextHMM ctx) {
		ContextHMM ret = ctx.copy();
		RealMatrix transition = MatrixUtils.createRealMatrix(ret.state_size, ret.state_size);
		transition = transition.scalarAdd(VERYSMALL);
		RealMatrix emission = MatrixUtils.createRealMatrix(ret.state_size, ret.symbol_size);
		emission = emission.scalarAdd(VERYSMALL);
		RealVector initial = MatrixUtils.createRealVector(new double[ret.state_size]);
		initial = initial.mapAddToSelf(VERYSMALL);
		for (int i = 0; i < seq.length; i++) {
			int[] path = decodingV2(seq[i], ret);
			double iCount = initial.getEntry(path[0]) + 1;
			initial.setEntry(path[0], iCount);
			double[] symb = seq[i].toArray();
			for (int j = 0; j < path.length - 1; j++) {
				counting(transition, emission, Double.valueOf(symb[j]).intValue(), new int[] { path[j], path[j + 1] });
			}
			counting(transition, emission, Double.valueOf(symb[path.length - 1]).intValue(),
					new int[] { path[path.length - 1] });
		}
		RealVector iden = MatrixUtils.createRealVector(new double[ret.state_size]);
		iden.set(1);
		for (int i = 0; i < transition.getRowDimension(); i++) {
			RealVector v = transition.getRowVector(i);
			double deno = iden.dotProduct(v);
			v = v.mapDivideToSelf(deno);
			transition.setRowVector(i, v);
		}

		double all_init = iden.dotProduct(initial);
		initial = initial.mapDivideToSelf(all_init);

		iden = MatrixUtils.createRealVector(new double[ret.symbol_size]);
		iden.set(1);
		for (int i = 0; i < emission.getRowDimension(); i++) {
			RealVector v = emission.getRowVector(i);
			double deno = iden.dotProduct(v);
			v = v.mapDivideToSelf(deno);
			emission.setRowVector(i, v);
		}

		ret.initialStates = initial;
		ret.transitionProbs = transition;
		ret.emissionProbs = emission;
		return ret;
	}

	static void counting(RealMatrix transition, RealMatrix emission, int symbol, int[] state) {
		int i = state[0]; // from state
		double e = emission.getEntry(i, symbol) + 1;
		emission.setEntry(i, symbol, e);
		if (state.length > 1) {
			int j = state[1];
			double t = transition.getEntry(i, j) + 1;
			transition.setEntry(i, j, t);
		}
	}

	/**
	 * <pre>
	 * α_i(t) = π_i * b_i(o_1)
	 * α_j(t+1) = b_j(o_t+1) * ∑1_N a_i(t) * a_ij
	 * </pre>
	 */
	static void alpha(RealVector sequence, ContextHMM ctx) {
		// initialize alpha(0) probs
		double[] o = sequence.toArray();

		RealVector a0 = ctx.emissionProbs.getColumnVector(Double.valueOf(o[0]).intValue());
		RealVector v = ctx.initialStates.ebeMultiply(a0);
		ctx.alpha.setColumnVector(0, v);

		for (int i = 1; i < o.length; i++) {
			RealVector last = ctx.alpha.getColumnVector(i - 1);
			int symbol = Double.valueOf(o[i]).intValue();
			v = ctx.transitionProbs.preMultiply(last);
			RealVector v1 = ctx.emissionProbs.getColumnVector(symbol);
			v = v1.ebeMultiply(v);
			ctx.alpha.setColumnVector(i, v);
		}
	}

	static void alphaV2(RealVector sequence, ContextHMM ctx) {
		double[] o = sequence.toArray();

		int symbol = Double.valueOf(o[0]).intValue();
		double[] a0 = ctx.emissionProbs.getColumnVector(symbol).toArray();
		double[] v0 = ctx.initialStates.toArray();
		Double[] v = ContextHMM.elnproduct(a0, v0);
		RealVector vv = MatrixUtils.createRealVector(ArrayUtils.toPrimitive(v));
		ctx.alpha.setColumnVector(0, vv);

		for (int i = 1; i < o.length; i++) {
			vv = ctx.alpha.getColumnVector(i - 1);
			double[] last = vv.toArray();
			symbol = Double.valueOf(o[i]).intValue();
			Double[] vvv = new Double[ctx.state_size];
			for (int to = 0; to < ctx.state_size; to++) {
				double[] a = new double[ctx.state_size];
				for (int from = 0; from < ctx.state_size; from++) {
					double tr = ctx.transitionProbs.getEntry(from, to);
					tr = ContextHMM.eln(tr);
					a[from] = last[from] + tr;
				}
				vvv[to] = ContextHMM.sumAtLogSpace(a);
				double outP = ctx.emissionProbs.getEntry(to, symbol);
				outP = ContextHMM.eln(outP);
				vvv[to] += outP;
			}
			vv = MatrixUtils.createRealVector(ArrayUtils.toPrimitive(vvv));
			ctx.alpha.setColumnVector(i, vv);
		}

	}

	/**
	 * Forward Algorithm
	 */
	static double evaluation(RealVector sequence, ContextHMM ctx) {
		ContextHMM model = ctx.copy();
		model.alpha = MatrixUtils.createRealMatrix(model.state_size, sequence.getDimension());
		alpha(sequence, model);
		RealVector iden = MatrixUtils.createRealVector(new double[model.state_size]);
		iden.set(1);
		RealVector v = model.alpha.getColumnVector(model.batch_size - 1);
		return iden.dotProduct(v);
	}

	/**
	 * Forward Algorithm at log space
	 */
	static double evaluationV2(RealVector sequence, ContextHMM ctx) {
		ContextHMM model = ctx.copy();
		model.alpha = MatrixUtils.createRealMatrix(model.state_size, sequence.getDimension());
		alphaV2(sequence, model);
		RealVector v = model.alpha.getColumnVector(model.batch_size - 1);
		return ContextHMM.eexp(ContextHMM.sumAtLogSpace(v.toArray()));
	}

	static int[] decodingV2(RealVector sequence, ContextHMM ctx) {
		double[] seq = sequence.toArray();
		ContextHMM model = ctx.copy();

		RealMatrix vb = MatrixUtils.createRealMatrix(model.state_size, seq.length);
		RealMatrix traceback = MatrixUtils.createRealMatrix(model.state_size, seq.length);
		int symbol = Double.valueOf(seq[0]).intValue();
		for (int from = 0; from < ctx.state_size; from++) {
			double pi = ContextHMM.eln(ctx.initialStates.getEntry(from));
			double em = ContextHMM.eln(ctx.emissionProbs.getEntry(from, symbol));
			Double p = pi + em;
			vb.setEntry(from, 0, p);
		}
		for (int t = 1; t < seq.length; t++) {
			symbol = Double.valueOf(seq[t]).intValue();
			for (int to = 0; to < ctx.state_size; to++) {
				double maxp = -1000d;
				int s = 0;
				for (int from = 0; from < ctx.state_size; from++) {
					double lastp = vb.getEntry(from, t - 1);
					double tr = ctx.transitionProbs.getEntry(from, to);
					tr = tr <= 0 ? 0 : FastMath.log(tr);
					double op = ctx.emissionProbs.getEntry(to, symbol);
					op = op <= 0 ? 0 : FastMath.log(op);
					double p = lastp + tr + op;
					if (p > maxp) {
						s = from;
						maxp = p;
					}
				}
				vb.setEntry(to, t, maxp);
				traceback.setEntry(to, t, s);
			}
		}

		int[] ret = new int[sequence.getDimension()];
		double maxp = -1000d;
		for (int s = 0; s < ctx.state_size; s++) {
			if (maxp < vb.getEntry(s, seq.length - 1)) {
				maxp = vb.getEntry(s, seq.length - 1);
				ret[seq.length - 1] = s;
			}
		}
		for (int t = seq.length - 1; t > 0; t--) {
			int s = ret[t];
			ret[t - 1] = Double.valueOf(traceback.getEntry(s, t)).intValue();
		}
		return ret;
	}
}
