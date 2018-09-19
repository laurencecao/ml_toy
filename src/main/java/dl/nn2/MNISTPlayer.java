package dl.nn2;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import dataset.MNIST;
import dl.nn2.layer.ConvolutionalLayer;
import dl.nn2.layer.DenseLayer;
import dl.nn2.layer.PoolingLayer;
import dl.nn2.layer.SoftmaxLayer;
import dl.nn2.model.NNModel;

public class MNISTPlayer {

	public static void main(String[] args) {
		List<MNIST> train = MNIST.getTraining();
		List<MNIST> test = MNIST.getTesting();
		NNModel model = setUp();
		work(model, train, test);
	}

	static NNModel setUp() {
		// Xavier.debug = 0.1d;
		// NNModel.resetLogging();
		// NNModel.setLoggingDebugMode(true);
		// TracedComputation.debugLevel(2);
		// RMSPropOptimizer.debugLevel(2);

		ConvolutionalLayer l1 = new ConvolutionalLayer(new int[] { 32, 32 }, 5, 1, 6, 1, 0, null, "L1");
		PoolingLayer l2 = new PoolingLayer(2, null, "L2");
		ConvolutionalLayer l3 = new ConvolutionalLayer(new int[] { 14, 14 }, 5, 6, 16, 1, 0, null, "L3");
		PoolingLayer l4 = new PoolingLayer(2, new int[] { 1, -1 }, "L4");
		DenseLayer l5 = new DenseLayer(400, 120, "L5");
		DenseLayer l6 = new DenseLayer(120, 84, "L6");
		SoftmaxLayer l7 = new SoftmaxLayer(84, 10, "L7");
		// RealMatrix w = MatrixUtils
		// .createRealMatrix(new double[][] { { 0.01, 0.1, 0.1 }, { 0.1, 0.2, 0.3 }, {
		// 0.1, 0.2, 0.3 } })
		// .transpose();
		// l1.updateWeights(new MatrixDataEdge("initial", w));
		// System.out.println(l1.getWeights().pretty());

		// l0.setBiased(0.1d);
		// l1.setBiased(0.1d);

		NNModel model = new NNModel(0.001d, 0.95d);
		model.setMinimumError(0.001d);
		// model.addLayer(l0);
		model.addLayer(l1);
		model.addLayer(l2);
		model.addLayer(l3);
		model.addLayer(l4);
		model.addLayer(l5);
		model.addLayer(l6);
		model.addLayer(l7);
		model.setLossName("crossentropy");
		// model.setOptimizer(new SimpleGradientDescend(0.5d, 0.95d));
		System.out.println(model.debugInfo());
		return model;
	}

	static void pickup(List<MNIST> data, List<RealMatrix> x, List<RealMatrix> y) {
		for (MNIST m : data) {
			int[][] d = m.getImage();
			RealMatrix dd = MatrixUtils.createRealMatrix(d.length, d[0].length);
			for (int i = 0; i < d.length; i++) {
				for (int j = 0; j < d[i].length; j++) {
					dd.setEntry(i, j, d[i][j]);
				}
			}
			x.add(dd);
			y.add(MatrixUtils.createRealMatrix(new double[][] { { m.getLabel() } }));
		}
	}

	static void work(NNModel model, List<MNIST> x, List<MNIST> t) {
		List<RealMatrix> train_x = new ArrayList<>();
		List<RealMatrix> train_y = new ArrayList<>();
		List<RealMatrix> test_x = new ArrayList<>();
		List<RealMatrix> test_y = new ArrayList<>();
		pickup(x, train_x, train_y);
		pickup(t, test_x, test_y);

		double loss = model.learning(0.001, train_x, train_y, 100, 500);
		int error = 0;
		int total = 0;
		List<RealMatrix> y = model.predict(test_x);
		for (int i = 0; i < t.size(); i++) {
			RealMatrix m = y.get(i);
			int Y = m.getColumnVector(0).getMaxIndex();
			m = test_y.get(i);
			int T = m.getColumnVector(0).getMaxIndex();
			System.out.println(Y + " ----> " + T);
			error += Y == T ? 0 : 1;
			total += 1;
		}

		System.out.println("Total Loss: " + (loss));
		System.out.println("Total Lost Count: " + error);
		System.out.println("Total Accuracy: " + (1.0d * (total - error) / total));
	}
}
