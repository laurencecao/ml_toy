package dl.nn2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import dataset.Moon;
import dl.nn2.layer.SoftmaxLayer;
import dl.nn2.model.NNModel;

public class MoonPlayer {

	public static void main(String[] args) {
		NNModel model = setUp();
		// List<RealMatrix>[] lst = demo();
		List<RealMatrix>[] lst = format(Moon.getData(), Moon.getLabel());
		work(model, lst[0], lst[1]);
	}

	@SuppressWarnings("unchecked")
	static List<RealMatrix>[] format(double[][] data, int[] label) {
		List<RealMatrix> x = new ArrayList<RealMatrix>();
		for (int i = 0; i < data.length; i++) {
			double[] d = data[i];
			RealMatrix m = MatrixUtils.createRealMatrix(d.length, 1);
			m.setColumn(0, d);
			x.add(m);
		}
		List<RealMatrix> y = onehot(label);
		return new List[] { x, y };
	}

	@SuppressWarnings("unchecked")
	static List<RealMatrix>[] demo() {
		List<RealMatrix> x = new ArrayList<>();
		RealMatrix m = null;
		m = MatrixUtils.createRealMatrix(new double[][] { { 0.1, 0.5 } }).transpose();
		x.add(m);
		m = MatrixUtils.createRealMatrix(new double[][] { { 1.1, 2.3 } }).transpose();
		x.add(m);
		m = MatrixUtils.createRealMatrix(new double[][] { { -1.1, -2.3 } }).transpose();
		x.add(m);
		m = MatrixUtils.createRealMatrix(new double[][] { { -1.5, -2.5 } }).transpose();
		x.add(m);
		List<RealMatrix> y = new ArrayList<>();
		m = MatrixUtils.createRealMatrix(new double[][] { { 1, 0, 0 } }).transpose();
		y.add(m);
		m = MatrixUtils.createRealMatrix(new double[][] { { 0, 1, 0 } }).transpose();
		y.add(m);
		m = MatrixUtils.createRealMatrix(new double[][] { { 0, 0, 1 } }).transpose();
		y.add(m);
		m = MatrixUtils.createRealMatrix(new double[][] { { 0, 0, 1 } }).transpose();
		y.add(m);
		return new List[] { x, y };
	}

	static void work(NNModel model, List<RealMatrix> x, List<RealMatrix> t) {
		double loss = model.learning(0.001, x, t, 100, 5000);

		int error = 0;
		int total = 0;
		List<RealMatrix> y = model.predict(x);
		for (int i = 0; i < t.size(); i++) {
			RealMatrix m = y.get(i);
			int Y = m.getColumnVector(0).getMaxIndex();
			m = t.get(i);
			int T = m.getColumnVector(0).getMaxIndex();
			m = x.get(i);
			String dmsg = Arrays.toString(m.getColumn(0));
			System.out.println(dmsg + " = " + Y + " ----> " + T);
			error += Y == T ? 0 : 1;
			total += 1;
		}

		System.out.println("Total Loss: " + (loss));
		System.out.println("Total Lost Count: " + error);
		System.out.println("Total Accuracy: " + (1.0d * (total - error) / total));
	}

	static NNModel setUp() {
		// Xavier.debug = 0.1d;
		NNModel.resetLogging();
		// NNModel.setLoggingDebugMode(true);
		// TracedComputation.debugLevel(2);
		// RMSPropOptimizer.debugLevel(2);

		// DenseLayer l0 = new DenseLayer(2, 2, "Layer0");
		SoftmaxLayer l1 = new SoftmaxLayer(2, 2, "Layer1");
		// RealMatrix w = MatrixUtils
		// .createRealMatrix(new double[][] { { 0.01, 0.1, 0.1 }, { 0.1, 0.2, 0.3 }, {
		// 0.1, 0.2, 0.3 } })
		// .transpose();
		// l1.updateWeights(new MatrixDataEdge("initial", w));
		System.out.println(l1.getWeights().pretty());

		// l0.setBiased(0.1d);
		// l1.setBiased(0.1d);

		NNModel model = new NNModel(0.5d, 0.95d);
		model.setMinimumError(0.001d);
		// model.addLayer(l0);
		model.addLayer(l1);

		model.setLossName("crossentropy");
		// model.setOptimizer(new SimpleGradientDescend(0.5d, 0.95d));
		return model;
	}

	static List<RealMatrix> onehot(int[] label) {
		List<RealMatrix> ret = new ArrayList<>();
		int sz = Arrays.stream(label).max().getAsInt() + 1;
		for (int i = 0; i < label.length; i++) {
			RealMatrix v = MatrixUtils.createRealMatrix(sz, 1);
			v.setEntry(label[i], 0, 1);
			ret.add(v);
		}
		return ret;
	}

}
