package dl.hmm;

import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;

import dataset.NNDataset;

public class FourCoin {

	/**
	 * second order Hidden Markov Model
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		RealVector[] seq = NNDataset.getData(NNDataset.FOURCOINS);
		// foobar2();

		// state = {A, B}; symbol = {T, H}
		ContextHMM model = learning(seq, 2, 2);
		// System.out.println(model.initialStates);
		// System.out.println("transition table => " + model.transitionProbs);
		// System.out.println("emission table => " + model.emissionProbs);

		print(model);
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
		xi(d[0], ctx);
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

		alpha(d[0], ctx);
		beta(d[0], ctx);
		normalizationAB(ctx);
		gamma(d[0], ctx);
		xi(d[0], ctx);
		normalizationGX(ctx);
		System.out.println(ctx);
	}

	static ContextHMM initContext(RealVector[] sequence, int state_size, int symbol_size) {
		ContextHMM ret = new ContextHMM(sequence[0].getDimension(), state_size, symbol_size);
		ret.symbolCounter = MatrixUtils.createRealVector(new double[ret.symbol_size]);
		ret.initialStates = MatrixUtils.createRealVector(new double[ret.state_size]);
		ret.initialStates.set(FastMath.pow(ret.state_size, -1));
		ret.transitionProbs = initMatrix(ret.state_size, ret.state_size, true);
		ret.emissionProbs = initMatrix(ret.state_size, ret.symbol_size, true);
		ret.alpha = initMatrix(ret.state_size, ret.batch_size, false);
		ret.beta = initMatrix(ret.state_size, ret.batch_size, false);
		ret.gamma = initMatrix(ret.state_size, ret.batch_size, false);
		ret.xi = new RealMatrix[ret.batch_size - 1];
		for (int i = 0; i < ret.xi.length; i++) {
			ret.xi[i] = initMatrix(ret.state_size, ret.state_size, false);
		}
		return ret;
	}

	static RealMatrixChangingVisitor rnd = new RealMatrixChangingVisitor() {

		@Override
		public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
		}

		@Override
		public double visit(int row, int column, double value) {
			return ThreadLocalRandom.current().nextDouble();
		}

		@Override
		public double end() {
			return 0;
		}

	};

	static RealMatrix initMatrix(int row, int col, boolean initial) {
		RealMatrix ret = MatrixUtils.createRealMatrix(row, col);
		if (initial) {
			ret = randonmize(ret);
		}
		return ret;
	}

	// normalization by column
	static RealMatrix randonmize(RealMatrix m) {
		RealMatrix ret = m.copy();
		ThreadLocalRandom rnd = ThreadLocalRandom.current();
		int c = ret.getColumnDimension();
		int r = ret.getRowDimension();
		RealVector cTotal = MatrixUtils.createRealVector(new double[c]);
		cTotal.set(1);
		RealVector rTotal = MatrixUtils.createRealVector(new double[r]);
		rTotal.set(1);
		for (int i = 1; i < r; i++) {
			double tR = rTotal.getEntry(i);
			for (int j = 1; j < c; j++) {
				double tC = cTotal.getEntry(j);
				double v = 2;
				while (v >= tR && v >= tC) {
					v = rnd.nextDouble();
				}
				tC -= v;
				tR -= v;
				cTotal.setEntry(j, tC);
				ret.setEntry(i, j, v);
			}
			rTotal.setEntry(i, tR);
		}
		ret.setRowVector(0, rTotal);
		ret.setColumnVector(0, cTotal);
		RealVector id = MatrixUtils.createRealVector(new double[c]);
		id.set(1);
		id.setEntry(0, -1);
		double v = id.dotProduct(ret.getRowVector(0));
		ret.setEntry(0, 0, -1d * v);
		return ret;
	}

	/**
	 * Baum-Welch Algorithm
	 * 
	 * @param sequence
	 */
	static ContextHMM learning(RealVector[] sequence, int state, int symbol) {
		ContextHMM ctx = initContext(sequence, state, symbol);
		for (int i = 0; i < sequence.length; i++) {
			RealVector seq = sequence[i];
			eStep(seq, ctx);
			mStep(seq, ctx, i);
		}
		return ctx;
	}

	static void eStep(RealVector seq, ContextHMM ctx) {
		alpha(seq, ctx);
		beta(seq, ctx);
		// normalizationAB(ctx);
		gamma(seq, ctx);
		xi(seq, ctx);
		normalizationGX(ctx);
	}

	static void mStep(RealVector seq, ContextHMM ctx, int round) {
		estimatePI(seq, ctx, round);
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

		// for (int i = 0; i < ctx.gamma.getRowDimension(); i++) {
		// RealVector v = ctx.gamma.getRowVector(i);
		// v.mapDivideToSelf(id.getEntry(i));
		// ctx.gamma.setRowVector(i, v);
		// }
	}

	static void xi(RealVector sequence, ContextHMM ctx) {
		double[] o = sequence.toArray(); // turn count
		RealVector iden = MatrixUtils.createRealVector(new double[ctx.state_size]);
		iden.set(1);
		RealVector last = ctx.alpha.getColumnVector(ctx.batch_size - 1);
		double denominator = iden.dotProduct(last);
		double total = 0d;
		for (int t = 0; t < ctx.batch_size - 1; t++) {
			int symbol = Double.valueOf(o[t + 1]).intValue();
			RealVector a = ctx.alpha.getColumnVector(t);
			for (int n = 0; n < ctx.state_size; n++) {
				RealVector tr = ctx.transitionProbs.getColumnVector(n);
				double b = ctx.beta.getEntry(n, t + 1);
				double q = ctx.emissionProbs.getEntry(n, symbol);
				RealVector v = a.ebeMultiply(tr).mapMultiply(b).mapMultiply(q);
				ctx.xi[t].setRowVector(n, v);
				// total += iden.dotProduct(v);
			}
		}
		// total = FastMath.pow(total, -1);
		total = FastMath.pow(denominator, -1);
		for (int t = 0; t < ctx.xi.length; t++) {
			ctx.xi[t] = ctx.xi[t].scalarMultiply(total);
		}
	}

	static void estimatePI(RealVector sequence, ContextHMM ctx, int round) {
		// beta distribution estimation
		int symbol = Double.valueOf(sequence.getEntry(0)).intValue();
		double ct = ctx.symbolCounter.getEntry(symbol) + 1;
		ctx.symbolCounter.setEntry(symbol, ct);

		double total = 0d;
		for (int i = 0; i < ctx.initialStates.getDimension(); i++) {
			ct = ctx.symbolCounter.getEntry(i);
			double pH = 1d * (ct + 1) / (round + 2);
			ctx.initialStates.setEntry(i, pH);
			total += pH;
		}

		// normalization
		ctx.initialStates.mapDivideToSelf(total);
	}

	static void estimateT(RealVector sequence, ContextHMM ctx) {
		RealVector iden = MatrixUtils.createRealVector(new double[ctx.batch_size - 1]);
		iden.set(1);
		int N = ctx.state_size;

		for (int i = 0; i < N; i++) {
			RealVector v = MatrixUtils.createRealVector(new double[ctx.state_size]);
			for (int t = 0; t < ctx.batch_size - 1; t++) {
				v = v.add(ctx.xi[t].getRowVector(i));
			}
			RealVector g = ctx.gamma.getRowVector(i).getSubVector(0, ctx.batch_size - 1);
			double deno = iden.dotProduct(g);
			deno = FastMath.pow(deno, -1);
			v = v.mapMultiply(deno);
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
		RealVector denominator = ctx.gamma.transpose().preMultiply(id);

		for (int i = 0; i < ctx.emissionProbs.getRowDimension(); i++) {
			double deno = denominator.getEntry(i);
			RealVector v = ctx.emissionProbs.getRowVector(i);
			v.mapDivideToSelf(deno);
			ctx.emissionProbs.setRowVector(i, v);
		}
	}

	static void normalizationAB(ContextHMM ctx) {
		// 1. alpha
		RealVector iden = MatrixUtils.createRealVector(new double[ctx.state_size]);
		iden.set(1); // identity

		for (int i = 0; i < ctx.alpha.getColumnDimension(); i++) {
			RealVector v = ctx.alpha.getColumnVector(i);
			double deno = iden.dotProduct(v);
			v.mapDivideToSelf(deno);
			ctx.alpha.setColumnVector(i, v);
		}

		// 2. beta
		for (int i = 0; i < ctx.beta.getColumnDimension(); i++) {
			RealVector v = ctx.beta.getColumnVector(i);
			double deno = iden.dotProduct(v);
			v.mapDivideToSelf(deno);
			ctx.beta.setColumnVector(i, v);
		}
	}

	static void normalizationGX(ContextHMM ctx) {
		// 1. gamma
		RealVector iden = MatrixUtils.createRealVector(new double[ctx.state_size]);
		iden.set(1); // identity

		for (int i = 0; i < ctx.gamma.getColumnDimension(); i++) {
			RealVector v = ctx.gamma.getColumnVector(i);
			double deno = iden.dotProduct(v);
			v.mapDivideToSelf(deno);
			ctx.gamma.setColumnVector(i, v);
		}

		// 2. xi
		for (int i = 0; i < ctx.xi.length; i++) {
			RealMatrix m = ctx.xi[i];
			double deno = iden.dotProduct(m.preMultiply(iden));
			m = m.scalarMultiply(FastMath.pow(deno, -1));
			ctx.xi[i] = m;
		}
	}

	/**
	 * Forward Algorithm
	 */
	static void evaluation(RealVector[] sequence, ContextHMM ctx) {

	}

	/**
	 * Forward-Backward Algorithm
	 */
	static void decoding(RealVector[] sequence, ContextHMM ctx) {

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

	// beta distribution, notion: α = 0.5, β = 0.5
	RealVector symbolCounter;
	RealVector initialStates; // π = state * 1
	RealMatrix transitionProbs; // t = state * state
	RealMatrix emissionProbs; // q = state * symbol

	RealMatrix alpha; // α = state * turn
	RealMatrix beta; // β = state * turn
	RealMatrix gamma; // γ = state * turn
	RealMatrix[] xi; // ξ = (state * state) [turn]

}
