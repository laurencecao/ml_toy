package dl.rbm;

import java.io.IOException;
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

import com.beust.jcommander.JCommander;

import dl.dataset.MovieLens;
import dl.dataset.NNDataset;
import dl.opt.DLParameter;

public class RBMPlayer {

	final static String MODEL_PATH = "/tmp/rate_rbm_%s.model";
	static double alpha = 0.1d;
	static double epi = 0.01d;
	static int dbgLoop = 100000;
	static int K = 2;
	static int H = 100;
	static UniformRandomGenerator urg = new UniformRandomGenerator(new JDKRandomGenerator());

	public static void main(String[] args) throws IOException {
		DLParameter param = new DLParameter();
		JCommander.newBuilder().addObject(param).build().parse(args);
		
		alpha = param.rate;
		epi = param.err;
		dbgLoop = param.debug;
		K = param.k;
		H = param.hidden;
		
		if (param.name.equalsIgnoreCase("digital")) {
			digital();
		}
		if (param.name.equalsIgnoreCase("rate")) {
			rate();
		}
	}

	static void digital() throws IOException {
//		alpha = 0.01d;
//		epi = 0.001d;
		SimpleRBM.SAVEIT = false;
		RealVector[] data = NNDataset.getData(NNDataset.DIGITAL);
		System.out.println("dataset size = " + data.length);
		for (int i = 0; i < 5; i++) {
			printOrigin(data[i], "" + (i + 1));
		}
		System.out.println("--------------- above is original ---------------------");
		int v = data[0].getDimension();
		SimpleRBM rbm = new SimpleRBM(v, H);
		rbm.K = K;
		rbm.rnd = new UncorrelatedRandomVectorGenerator(data.length, urg);
		training(rbm, data, dbgLoop);
		for (int i = 0; i < data.length; i++) {
			RealVector[] p = predict(rbm, data[i]);
			printDebugInfo(p, "" + i);
		}
		RealVector[] p = predict(rbm, MatrixUtils.createRealVector(new double[] { 0, 0, 0, 1, 1, 0 }));
		printDebugInfo(p, "aabb");
	}

	static void rate() throws IOException {
		SimpleRBM.SAVEIT = true;
		// alpha = 0.2d;
		// epi = 10;
		RealVector[] data = NNDataset.getData(NNDataset.MOVIELENS);
		System.out.println("dataset size = " + data.length);
		for (int i = 0; i < 5; i++) {
			printRate(data[i], "" + (i + 1));
		}
		System.out.println("--------------- above is original ---------------------");
		int v = data[0].getDimension();
		int h = H;
		SimpleRBM rbm = SimpleRBM.load(String.format(MODEL_PATH, h));
		if (rbm == null) {
			rbm = new SimpleRBM(v, h);
			rbm.K = K;
		}
		rbm.rnd = new UncorrelatedRandomVectorGenerator(data.length, urg);
		training(rbm, data, 1);
		for (int i = 0; i < 10; i++) {
			RealVector[] p = predict(rbm, data[i]);
			printRate(p[p.length - 1], "" + (i + 1));
		}
	}

	public static void training(SimpleRBM rbm, RealVector[] all_data, int debugLoop) throws IOException {
		int v = all_data[0].getDimension();
		int count = all_data.length;
		RealMatrix data = MatrixUtils.createRealMatrix(v, count);
		for (int i = 0; i < all_data.length; i++) {
			data.setColumn(i, all_data[i].toArray());
		}

		long ts = System.currentTimeMillis();
		double norm = epi * 10000;
		double rate = alpha;
		for (int i = 0; norm > epi; i++) {
			norm = CD_K(rbm, data, rate);
			if (i % debugLoop == 0) {
				ts = System.currentTimeMillis() - ts;
				System.out.println(
						"CD_" + rbm.K + "[" + debugLoop + "], r = " + rate + " elapsed: " + ts + "ms; norm = " + norm);
				// double r = rate;
				// rate *= 0.99d;
				// System.out.println("tuning learning rate " + r + " --> " +
				// rate);
				SimpleRBM.save(rbm, String.format(MODEL_PATH, rbm.hidden));
				ts = System.currentTimeMillis();
			}
			if (norm < epi) {
				break;
			}
		}
		System.out.println("Last Norm: " + norm);
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

	static double CD_K(SimpleRBM rbm, RealMatrix data, double rate) {
		RealMatrix data0 = addBiase(data);
		RealMatrix data1 = data0.copy();

		RealMatrix last_p = null;
		int count = data0.getColumnDimension();
		List<RealMatrix> delta = new ArrayList<RealMatrix>();
		for (int i = 0; i < rbm.K; i++) {
			// positive CD phase
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
			last_p = p1;
			data1 = p1.copy();
			data1.walkInOptimizedOrder(setState);
			data1.walkInRowOrder(setBit, 0, 0, 0, p1.getColumnDimension() - 1);
		}

		RealMatrix p = rbm.weights.multiply(last_p);
		p.walkInOptimizedOrder(activation);
		delta.add(p.multiply(last_p.transpose())); // add E(h|v)

		RealMatrix asso_0 = delta.get(0);
		RealMatrix asso_n = delta.get(delta.size() - 1);
		RealMatrix dw = asso_n.subtract(asso_0).scalarMultiply(rate / count);
		rbm.weights = rbm.weights.subtract(dw);

		return data0.subtract(last_p).getNorm();
	}

	static RealMatrix addBiase(RealMatrix d) {
		RealMatrix ret = d.createMatrix(d.getRowDimension() + 1, d.getColumnDimension());
		ret.walkInRowOrder(setBit, 0, 0, 0, ret.getColumnDimension() - 1);
		ret.setSubMatrix(d.getData(), 1, 0);
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

	static void printRate(RealVector v, String name) {
		double[] vv = v.toArray();
		StringBuilder sb = new StringBuilder();
		sb.append("user[").append(name).append("]: {   ");
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
