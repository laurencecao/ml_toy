package dl.hmm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.FastMath;

import dataset.NNDataset;

public class FourCoin {

	final static int TURN = 100;
	final static double DELTA_CHANGE = 0.001;

	/**
	 * second order Hidden Markov Model
	 * 
	 * 
	 * sigh! very sensitive to initial parameters
	 * 
	 * <pre>
	Initial CoinA => 0.5474899276452769
	Initial CoinB => 0.452510072354721
	Transition Table: 
	CoinA => CoinA: 0.5095629234833225
	CoinA => CoinB: 0.49043707651667773
	CoinB => CoinA: 0.5348077810413786
	CoinB => CoinB: 0.4651922189586202
	Emission Table: 
	CoinA => HEAD: 0.29013594445632124
	CoinA => TAIL: 0.709864055543679
	CoinB => HEAD: 0.5981318549577567
	CoinB => TAIL: 0.40186814504224344
	TTTTTTHHTT => 0.0019267967917331837
	TTTTTHHTHT => 0.0015116099142341382
	[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
	[0, 0, 0, 0, 0, 1, 1, 0, 1, 0]
	 * </pre>
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		RealVector[] seq = NNDataset.getData(NNDataset.FOURCOINS);
		// foobar2();
		// boolean b = true;
		// if (b){
		// System.exit(0);
		// }

		/**
		 * state = {A, B}; symbol = {T, H}
		 */
		ContextHMM model = learning(seq, 2, 2);
		System.out.println("After training .......... ");
		print(model);

		// "TTTTTTHHTT" -> 8
		RealVector s1 = MatrixUtils.createRealVector(new double[] { 0, 0, 0, 0, 0, 0, 1, 1, 0, 0 });
		// "TTTTTHHTHT" -> 1
		RealVector s2 = MatrixUtils.createRealVector(new double[] { 0, 0, 0, 0, 0, 1, 1, 0, 1, 0 });

		System.out.println("TTTTTTHHTT => " + evaluation(s1, model));
		System.out.println("TTTTTHHTHT => " + evaluation(s2, model));

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
		d[0] = MatrixUtils.createRealVector(new double[] { 0, 1, 0 });
		ContextHMM ctx = initContext(d, 2, 2);
		ctx.initialStates.setEntry(0, 0.99d);
		ctx.initialStates.setEntry(1, 0.01d);

		ctx.transitionProbs.setEntry(0, 0, 0.99d);
		ctx.transitionProbs.setEntry(0, 1, 0.01d);
		ctx.transitionProbs.setEntry(1, 0, 0.01d);
		ctx.transitionProbs.setEntry(1, 1, 0.99d);

		ctx.emissionProbs.setEntry(0, 0, 0.8d);
		ctx.emissionProbs.setEntry(0, 1, 0.2d);
		ctx.emissionProbs.setEntry(1, 0, 0.1d);
		ctx.emissionProbs.setEntry(1, 1, 0.9d);

		alpha(d[0], ctx);
		beta(d[0], ctx);
		gamma(d[0], ctx);
		ksi(d[0], ctx);
		System.out.println(ctx);
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

		for (int i = 0; i < 2; i++) {
			alpha(d[0], ctx);
			beta(d[0], ctx);
			// normalizationAB(ctx);
			gamma(d[0], ctx);
			ksi(d[0], ctx);
			// normalizationGX(ctx);
			estimatePI(d[0], ctx);
			estimateT(d[0], ctx);
			estimateQ(d[0], ctx);

			print(ctx);
		}

		System.out.println(Arrays.toString(d[0].toArray()) + " => " + evaluation(d[0], ctx));

		int[] symbol = null;
		symbol = decoding(d[0], ctx);
		System.out.println(Arrays.toString(symbol));
	}

	static ContextHMM initContext(RealVector[] sequence, int state_size, int symbol_size) {
		ContextHMM ret = new ContextHMM(sequence[0].getDimension(), state_size, symbol_size);
		RealMatrix init = initMatrix(ret.state_size, 1, true);
		ret.initialStates = normalizeByColumn(init).getColumnVector(0);
		ret.initialStates.set(1d / state_size);
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

	/**
	 * Baum-Welch Algorithm
	 * 
	 * @param sequence
	 */
	static ContextHMM learning(RealVector[] sequence, int state, int symbol) {
		List<ContextHMM> est = new ArrayList<ContextHMM>();
		ContextHMM origin = initContext(sequence, state, symbol);
		System.out.println("Before training ............ ");
		print(origin);

		double lastTurnLike = 1000000;
		ContextHMM ret = origin.copy();
		for (int t = 0; t < TURN; t++) {
			for (int i = 0; i < sequence.length; i++) {
				ContextHMM ctx = ret.copy();
				RealVector seq = sequence[i];
				double like = FastMath.log(evaluation(seq, ctx));
				eStep(seq, ctx);
				mStep(seq, ctx);
				ctx.deltaLoglike = FastMath.abs(FastMath.log(evaluation(seq, ctx)) - like);
				est.add(ctx);
				// System.out.println("............ turn " + i);
				// print(ctx);
			}

			SummaryStatistics mean = new SummaryStatistics();
			ContextHMM c = ret.copy();
			RealMatrix iden;
			c.initialStates.set(0);
			iden = MatrixUtils.createRealMatrix(c.state_size, c.state_size);
			c.transitionProbs = iden.multiply(c.transitionProbs);
			iden = MatrixUtils.createRealMatrix(c.state_size, c.state_size);
			c.emissionProbs = iden.multiply(c.emissionProbs);
			for (int i = 0; i < est.size(); i++) {
				// mean(t+1) = mean(t) + [X(t+1) - mean(t)] / (t+1)
				ContextHMM ctx = est.get(i);
				double deno = FastMath.pow(i + 1, -1);

				// pi
				RealVector v = ctx.initialStates.subtract(c.initialStates).mapMultiply(deno);
				c.initialStates = c.initialStates.add(v);

				// transition
				RealMatrix vv = ctx.transitionProbs.subtract(c.transitionProbs).scalarMultiply(deno);
				c.transitionProbs = c.transitionProbs.add(vv);

				// emission
				vv = ctx.emissionProbs.subtract(c.emissionProbs).scalarMultiply(deno);
				c.emissionProbs = c.emissionProbs.add(vv);

				mean.addValue(ctx.deltaLoglike);
			}

			c.deltaLoglike = mean.getMean();
			ret = c;

			if (c.deltaLoglike - lastTurnLike < DELTA_CHANGE) {
				break;
			}
			lastTurnLike = c.deltaLoglike;
		}
		return ret.copy();
	}

	static void eStep(RealVector seq, ContextHMM ctx) {
		alpha(seq, ctx);
		beta(seq, ctx);
		// normalizationAB(ctx);
		gamma(seq, ctx);
		ksi(seq, ctx);
	}

	static void mStep(RealVector seq, ContextHMM ctx) {
		estimatePI(seq, ctx);
		estimateT(seq, ctx);
		estimateQ(seq, ctx);
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
	 * <pre>
	 * β_i(T) = 1
	 * β_i(t) = ∑1_N β_j(t+1) * a_ij * b_j(o_t+1)
	 * </pre>
	 */
	static void beta(RealVector sequence, ContextHMM ctx) {
		double[] o = sequence.toArray();

		// initialization β_i(T)
		RealVector v = MatrixUtils.createRealVector(new double[ctx.state_size]);
		v.set(1);
		ctx.beta.setColumnVector(o.length - 1, v);

		for (int i = o.length - 2; i >= 0; i--) {
			RealVector beta = ctx.beta.getColumnVector(i + 1);
			RealMatrix tr = ctx.transitionProbs.transpose();
			int symbol = Double.valueOf(o[i + 1]).intValue();
			RealVector q = ctx.emissionProbs.getColumnVector(symbol);
			v = tr.preMultiply(beta.ebeMultiply(q));
			ctx.beta.setColumnVector(i, v);
		}
	}

	/**
	 * <pre>
	 * 
	 * </pre>
	 * 
	 * @param sequence
	 * @param ctx
	 */
	static void gamma(RealVector sequence, ContextHMM ctx) {
		RealVector id = MatrixUtils.createRealVector(new double[ctx.state_size]);

		for (int i = 0; i < ctx.batch_size; i++) {
			id.set(1);
			RealVector a = ctx.alpha.getColumnVector(i);
			RealVector b = ctx.beta.getColumnVector(i);
			RealVector v = a.ebeMultiply(b);
			double deno = id.dotProduct(v);
			v = v.mapDivideToSelf(deno);
			ctx.gamma.setColumnVector(i, v);
		}
	}

	static void ksi(RealVector sequence, ContextHMM ctx) {
		double[] o = sequence.toArray(); // turn count
		RealVector iden = MatrixUtils.createRealVector(new double[ctx.state_size]);
		iden.set(1);
		for (int t = 0; t < ctx.batch_size - 1; t++) {
			int symbol = Double.valueOf(o[t + 1]).intValue();
			RealVector a = ctx.alpha.getColumnVector(t);
			double deno = 0d;
			for (int n = 0; n < ctx.state_size; n++) {
				RealVector tr = ctx.transitionProbs.getColumnVector(n);
				double b = ctx.beta.getEntry(n, t + 1);
				double q = ctx.emissionProbs.getEntry(n, symbol);
				RealVector v = a.ebeMultiply(tr).mapMultiply(b).mapMultiply(q);
				ctx.ksi[t].setRowVector(n, v);
				deno += iden.dotProduct(v);
			}
			deno = FastMath.pow(deno, -1);
			ctx.ksi[t] = ctx.ksi[t].scalarMultiply(deno).transpose();
		}
	}

	static void estimatePI(RealVector sequence, ContextHMM ctx) {
		RealVector v = ctx.gamma.getColumnVector(0);
		ctx.initialStates = v;
	}

	static void estimateT(RealVector sequence, ContextHMM ctx) {
		RealVector iden = MatrixUtils.createRealVector(new double[ctx.batch_size - 1]);
		iden.set(1);
		int N = ctx.state_size;
		RealVector id = MatrixUtils.createRealVector(new double[ctx.state_size]);
		id.set(1);

		for (int i = 0; i < N; i++) {
			RealVector v = MatrixUtils.createRealVector(new double[ctx.state_size]);
			for (int t = 0; t < ctx.batch_size - 1; t++) {
				v = v.add(ctx.ksi[t].getRowVector(i));
			}
			RealVector g = ctx.gamma.getRowVector(i).getSubVector(0, ctx.batch_size - 1);
			double deno = iden.dotProduct(g);
			deno = FastMath.pow(deno, -1);
			v = v.mapMultiply(deno);
			// deno = id.dotProduct(v);
			// deno = FastMath.pow(deno, -1);
			// v = v.mapMultiply(deno);
			ctx.transitionProbs.setRowVector(i, v);
		}
	}

	static void estimateQ(RealVector sequence, ContextHMM ctx) {
		double[] o = sequence.toArray();
		RealMatrix iden = MatrixUtils.createRealMatrix(ctx.symbol_size, ctx.batch_size);
		for (int t = 0; t < o.length; t++) {
			int symbol = Double.valueOf(o[t]).intValue();
			iden.setEntry(symbol, t, 1);
		}

		RealMatrix g = ctx.gamma;
		ctx.emissionProbs = iden.multiply(g.transpose()).transpose();

		RealVector id = MatrixUtils.createRealVector(new double[ctx.batch_size]);
		id.set(1);
		RealVector denominator = g.transpose().preMultiply(id);

		for (int i = 0; i < ctx.emissionProbs.getRowDimension(); i++) {
			double deno = denominator.getEntry(i);
			RealVector v = ctx.emissionProbs.getRowVector(i);
			v.mapDivideToSelf(deno);
			ctx.emissionProbs.setRowVector(i, v);
		}
	}

	static void normalizationAB(ContextHMM ctx) {
		RealVector iden = MatrixUtils.createRealVector(new double[ctx.state_size]);
		iden.set(1); // identity

		for (int i = 0; i < ctx.alpha.getColumnDimension(); i++) {
			RealVector a = ctx.alpha.getColumnVector(i);
			RealVector b = ctx.beta.getColumnVector(i);
			double deno = iden.dotProduct(a);
			a = a.mapDivideToSelf(deno);
			b = b.mapDivideToSelf(deno);
			ctx.alpha.setColumnVector(i, a);
			ctx.beta.setColumnVector(i, b);
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
				double last_op = 0d;
				double last_idx = 0d;
				// any_state -> s
				RealVector tr = model.transitionProbs.getColumnVector(s);
				// backtrace probs
				RealVector btProbs = tr.ebeMultiply(v);
				for (int j = 0; j < btProbs.getDimension(); j++) {
					// j is from_state;
					double p = btProbs.getEntry(j);
					if (last_p < p) { // p max but op maybe not
						// s is current_state
						double op = p * output.getEntry(s);
						last_op = op;
						last_p = p; // max
						last_idx = j;
					}
				}
				viterbi.setEntry(s, i, last_op);
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

class ContextHMM {

	final int batch_size;
	final int state_size;
	final int symbol_size;

	ContextHMM(int batch, int state, int symbol) {
		this.batch_size = batch;
		this.state_size = state;
		this.symbol_size = symbol;
	}

	RealVector initialStates; // π = state * 1
	RealMatrix transitionProbs; // t = state * state
	RealMatrix emissionProbs; // q = state * symbol

	RealMatrix alpha; // α = state * turn
	RealMatrix beta; // β = state * turn
	RealMatrix gamma; // γ = state * turn
	RealMatrix[] ksi; // ξ = (state * state) [turn]

	double deltaLoglike;

	public ContextHMM copy() {
		ContextHMM ret = new ContextHMM(batch_size, state_size, symbol_size);
		ret.initialStates = initialStates.copy();
		ret.transitionProbs = transitionProbs.copy();
		ret.emissionProbs = emissionProbs.copy();
		ret.alpha = alpha.copy();
		ret.beta = beta.copy();
		ret.gamma = gamma.copy();
		ret.ksi = new RealMatrix[ksi.length];
		for (int i = 0; i < ksi.length; i++) {
			ret.ksi[i] = ksi[i].copy();
		}
		return ret;
	}

}
