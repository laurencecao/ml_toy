package dl.hmm;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import org.apache.commons.math3.linear.RealVector;

import dataset.NNDataset;

public class FourCoinViterbi {

	final static int TURN = 10;
	final static double EPSILON = 0.001d;
	final static int DEBUG = 10;

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

		// "HHHHHHHHTHHHHHHHTHTH" -> 2
		RealVector s1 = MatrixUtils
				.createRealVector(new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1 });
		// "TTTTTTTTTHTTHHHTTTTT" -> 1
		RealVector s2 = MatrixUtils
				.createRealVector(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0 });

		System.out.println("HHHHHHHHTHHHHHHHTHTH => " + evaluation(s1, model));
		System.out.println("TTTTTTTTTHTTHHHTTTTT => " + evaluation(s2, model));

		int[] symbol = null;
		symbol = decoding(s1, model);
		System.out.println(Arrays.toString(symbol));
		symbol = decoding(s2, model);
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
		symbol = decoding(s2, model);
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

		int[] demo = decoding(d[0], ctx);
		System.out.println(Arrays.toString(demo));

		for (int i = 0; i < 100; i++) {
			ContextHMM c = mStep(d, ctx);
			demo = decoding(d[0], c);
			System.out.println("decoding at [" + i + "] --> " + Arrays.toString(demo));
			ctx = c;

			if (i % 10 == 0) {
				print(c);
			}
		}
		print(ctx);

		System.out.println(Arrays.toString(d[0].toArray()) + " => " + evaluation(d[0], ctx));

		int[] symbol = null;
		symbol = decoding(d[0], ctx);
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
		double n1 = c1.transitionProbs.subtract(c2.transitionProbs).getNorm();
		double n2 = c1.emissionProbs.subtract(c2.emissionProbs).getNorm();
		return (n1 + n2) / 2;
	}

	/**
	 * Baum-Welch Algorithm
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
		transition = transition.scalarAdd(0.01d);
		RealMatrix emission = MatrixUtils.createRealMatrix(ret.state_size, ret.symbol_size);
		emission = emission.scalarAdd(0.01d);
		RealVector initial = MatrixUtils.createRealVector(new double[ret.state_size]);
		initial = initial.mapAddToSelf(0.01d);
		for (int i = 0; i < seq.length; i++) {
			int[] path = decoding(seq[i], ret);
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
	 * Viterbi Algorithm
	 */
	static int[] decoding(RealVector sequence, ContextHMM ctx) {
		double[] seq = sequence.toArray();
		ContextHMM model = ctx.copy();
		RealMatrix viterbi = MatrixUtils.createRealMatrix(model.state_size, seq.length);
		RealMatrix bt = viterbi.copy();
		int symbol = Double.valueOf(seq[0]).intValue();
		RealVector output = model.emissionProbs.getColumnVector(symbol);
		viterbi.setColumnVector(0, model.initialStates.ebeMultiply(output));

		for (int i = 1; i < seq.length; i++) {
			symbol = Double.valueOf(seq[i]).intValue();
			output = model.emissionProbs.getColumnVector(symbol);
			RealVector v = viterbi.getColumnVector(i - 1);

			for (int s = 0; s < model.state_size; s++) {
				double last_p = 0d;
				double last_idx = 0d;
				// any_state -> s
				RealVector tr = model.transitionProbs.getColumnVector(s);
				// backtrace probs
				RealVector btProbs = tr.ebeMultiply(v).ebeMultiply(output);
				for (int j = 0; j < btProbs.getDimension(); j++) {
					// j is from_state;
					double p = btProbs.getEntry(j);
					if (last_p < p) { // p max but op maybe not
						last_p = p; // max
						last_idx = j;
					}
				}
				viterbi.setEntry(s, i, last_p);
				bt.setEntry(s, i, last_idx);
			}
		}

		RealVector last = viterbi.getColumnVector(viterbi.getColumnDimension() - 1);
		double last_p = 0d;
		int last_idx = 0;
		for (int s = 0; s < last.getDimension(); s++) {
			if (last_p < last.getEntry(s)) {
				last_p = last.getEntry(s);
				last_idx = s;
			}
		}

		int[] ret = new int[sequence.getDimension()];
		ret[ret.length - 1] = last_idx;
		for (int i = ret.length - 2; i >= 0; i--) {
			int idx = i + 1;
			ret[i] = Double.valueOf(bt.getEntry(ret[idx], idx)).intValue();
		}

		return ret;
	}

}
