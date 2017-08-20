package dl.rbm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.UncorrelatedRandomVectorGenerator;
import org.apache.commons.math3.random.UniformRandomGenerator;
import org.apache.commons.math3.util.FastMath;

import dl.dataset.MovieLens;
import dl.dataset.NNDataset;

public class RBMPlayer {

	static double alpha = 0.01d;
	static double epi = 0.0001d;
	static UniformRandomGenerator urg = new UniformRandomGenerator(new JDKRandomGenerator());

	public static void main(String[] args) {
		// rate();
		digital();
	}

	static void digital() {
		RealVector[] data = NNDataset.getData(NNDataset.DIGITAL);
		System.out.println("dataset size = " + data.length);
		for (int i = 0; i < 5; i++) {
			printOrigin(data[i], "" + (i + 1));
		}
		System.out.println("--------------- above is original ---------------------");
		int v = data[0].getDimension();
		SimpleRBM rbm = new SimpleRBM(v, 2);
		rbm.rnd = new UncorrelatedRandomVectorGenerator(data.length, urg);
		training(rbm, data);
		for (int i = 0; i < data.length; i++) {
			RealVector[] p = predict(rbm, data[i]);
			printDebugInfo(p, "" + i);
		}
		RealVector[] p = predict(rbm, MatrixUtils.createRealVector(new double[] { 0, 0, 0, 1, 1, 0 }));
		printDebugInfo(p, "aabb");
	}

	static void rate() {
		RealVector[] data = NNDataset.getData(NNDataset.MOVIELENS);
		System.out.println("dataset size = " + data.length);
		for (int i = 0; i < 5; i++) {
			printData(data[i], i + 1);
		}
		System.out.println("--------------- above is original ---------------------");
		int v = data[0].getDimension();
		SimpleRBM rbm = new SimpleRBM(v, 200);
		rbm.rnd = new UncorrelatedRandomVectorGenerator(data.length, urg);
		training(rbm, data);
		for (int i = 0; i < 2; i++) {
			// RealVector p = predict(rbm, data[i]);
			// printData(p, i + 1);
		}
	}

	public static void training(SimpleRBM rbm, RealVector[] all_data) {
		int v = all_data[0].getDimension();
		int count = all_data.length;
		RealMatrix data = MatrixUtils.createRealMatrix(v, count);
		for (int i = 0; i < all_data.length; i++) {
			data.setColumn(i, all_data[i].toArray());
		}

		long ts;
		int K = 10;
		double norm = 10;
		for (int i = 0; norm > epi; i++) {
			ts = System.currentTimeMillis();
			norm = CD_K(K, rbm, data);
			ts = System.currentTimeMillis() - ts;
			if (i % 1000 == 0) {
				System.out.println("CD_" + K + " elapsed: " + ts + "ms; norm = " + norm);
			}
			if (norm < epi) {
				break;
			}
		}
		System.out.println("Last Norm: " + norm);
	}

	public static RealVector predict2(SimpleRBM rbm, RealVector d) {
		RealMatrix data = MatrixUtils.createRealMatrix(d.getDimension(), 1);
		data.setColumn(0, d.toArray());
		RealMatrix data0 = addBiase(data);
		RealMatrix data1 = data0.copy();
		RealMatrix p0 = rbm.weights.multiply(data1);
		p0.walkInOptimizedOrder(activation);

		RealMatrix state = p0.copy();
		state.walkInOptimizedOrder(setState);

		state.getRowVector(0).set(1); // always biased = 1

		// negative CD phase
		RealMatrix p1 = rbm.weights.transpose().multiply(state);
		p1.walkInOptimizedOrder(activation);
		p1.walkInOptimizedOrder(setState);
		return p1.getColumnVector(0).getSubVector(1, data1.getRowDimension() - 1);
	}

	public static RealVector[] predict(SimpleRBM rbm, RealVector d) {
		RealMatrix data = MatrixUtils.createColumnRealMatrix(d.toArray());
		RealMatrix data0 = addBiase(data);
		RealMatrix p0 = rbm.weights.multiply(data0);
		p0.walkInOptimizedOrder(activation);

		RealMatrix hidden_state = p0.copy();
		hidden_state.walkInOptimizedOrder(setState);
		// always biased = 1
		hidden_state.walkInRowOrder(setBit, 0, 0, 0, hidden_state.getColumnDimension() - 1);

		RealMatrix p1 = rbm.weights.transpose().multiply(hidden_state);
		p1.walkInOptimizedOrder(activation);
		p1.walkInRowOrder(setBit, 0, 0, 0, p1.getColumnDimension() - 1);

		RealMatrix visible_state = p1.copy();
		visible_state.walkInOptimizedOrder(setState);

		return new RealVector[] { d, p0.getColumnVector(0).getSubVector(1, p0.getRowDimension() - 1),
				p1.getColumnVector(0).getSubVector(1, p1.getRowDimension() - 1),
				visible_state.getColumnVector(0).getSubVector(1, visible_state.getRowDimension() - 1) };
	}

	static double CD_K(int k, SimpleRBM rbm, RealMatrix data) {
		RealMatrix data0 = addBiase(data);
		RealMatrix data1 = data0.copy();

		int count = data0.getColumnDimension();
		List<RealMatrix> delta = new ArrayList<RealMatrix>();
		for (int i = 0; i < k; i++) {
			RealMatrix p0 = rbm.weights.multiply(data1);
			p0.walkInOptimizedOrder(activation);

			// add E(h|v) for calculate dw
			delta.add(p0.multiply(data1.transpose()));

			RealMatrix state = p0.copy();
			state.walkInOptimizedOrder(setState);
			// always biased = 1
			state.walkInRowOrder(setBit, 0, 0, 0, state.getColumnDimension() - 1);

			// negative CD phase
			RealMatrix p1 = rbm.weights.transpose().multiply(state);
			p1.walkInOptimizedOrder(activation);
			p1.walkInOptimizedOrder(setState);
			p1.walkInRowOrder(setBit, 0, 0, 0, p1.getColumnDimension() - 1);
			data1 = p1;
		}

		RealMatrix p = rbm.weights.multiply(data1);
		p.walkInOptimizedOrder(activation);
		delta.add(p.multiply(data1.transpose())); // add E(h|v)

		RealMatrix asso_0 = delta.get(0);
		RealMatrix asso_n = delta.get(delta.size() - 1);
		RealMatrix dw = asso_n.subtract(asso_0);
		rbm.weights = rbm.weights.subtract(dw.scalarMultiply(alpha / count));

		return data0.subtract(data1).getNorm();
	}

	static RealMatrix addBiase(RealMatrix d) {
		RealMatrix ret = d.createMatrix(d.getRowDimension() + 1, d.getColumnDimension());
		ret.walkInRowOrder(setBit, 0, 0, 0, ret.getColumnDimension() - 1);
		ret.setSubMatrix(d.getData(), 1, 0);
		return ret;
	}

	static RealMatrix randMatrix(UncorrelatedRandomVectorGenerator rnd, int r, int c) {
		RealMatrix ret = MatrixUtils.createRealMatrix(r, c);
		for (int j = 0; j < r; j++) {
			RealVector s = MatrixUtils.createRealVector(rnd.nextVector());
			ret.setRow(j, s.toArray());
		}
		return ret;
	}

	static void printDebugInfo(RealVector[] v, String name) {
		StringBuilder sb = new StringBuilder();
		sb.append("[").append(name).append("]: ");
		sb.append("original state --> ").append(Arrays.toString(v[0].toArray())).append(" ; \n");
		sb.append("[").append(name).append("]: ");
		sb.append("hidden state --> ").append(Arrays.toString(v[1].toArray())).append(" ; \n");
		sb.append("[").append(name).append("]: ");
		sb.append("visible probability--> ").append(Arrays.toString(v[2].toArray())).append(" ; \n");
		sb.append("[").append(name).append("]: ");
		sb.append("visible state --> ").append(Arrays.toString(v[3].toArray()));
		System.out.println(sb.toString());
	}

	static void printOrigin(RealVector v, String name) {
		StringBuilder sb = new StringBuilder();
		sb.append("[").append(name).append("]: ");
		sb.append("visible --> ").append(Arrays.toString(v.toArray()));
		System.out.println(sb.toString());
	}

	static void printData(RealVector v, int id) {
		double[] vv = v.toArray();
		StringBuilder sb = new StringBuilder();
		sb.append("user[").append(id).append("]: {   ");
		for (int i = 0; i < vv.length; i++) {
			if (vv[i] != 0) {
				sb.append(MovieLens.dictionary.get(i)).append(", ");
			}
		}
		sb.delete(sb.length() - 2, sb.length());
		sb.append(" }");
		System.out.println(sb.toString());
	}

	final static RealMatrixChangingVisitor activation = new RealMatrixChangingVisitor() {
		@Override
		public double end() {
			return 0;
		}

		@Override
		public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
		}

		@Override
		public double visit(int row, int column, double value) {
			return 1.0 / (1 + FastMath.exp(-1 * value));
		}
	};

	final static RealMatrixChangingVisitor setState = new RealMatrixChangingVisitor() {
		@Override
		public double end() {
			return 0;
		}

		@Override
		public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
		}

		@Override
		public double visit(int row, int column, double value) {
			return Math.random() < value ? 1 : 0;
		}
	};

	final static RealMatrixChangingVisitor setBit = new RealMatrixChangingVisitor() {
		@Override
		public double end() {
			return 0;
		}

		@Override
		public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
		}

		@Override
		public double visit(int row, int column, double value) {
			return 1d;
		}
	};

}
