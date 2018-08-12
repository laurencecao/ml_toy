package dl.nn2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import dl.nn2.layer.DenseLayer;
import dl.nn2.model.NNModel;

public class XOR {

	public static void main(String[] args) {
		long ts = System.currentTimeMillis();
		// RealMatrix x = MatrixUtils.createRealMatrix(new double[][] { { -1, -1 }, {
		// -1, 1 }, { 1, -1 }, { 1, 1 } })
		// .transpose();
		// RealMatrix y = MatrixUtils.createRealMatrix(new double[][] { { -1 }, { 1 }, {
		// 1 }, { -1 } }).transpose();
		RealMatrix x = MatrixUtils.createRealMatrix(new double[][] { { 1, 0 }, { 1, 1 }, { 0, 1 }, { 0, 0 } })
				.transpose();
		RealMatrix y = MatrixUtils.createRealMatrix(new double[][] { { 1 }, { 0 }, { 1 }, { 0 } }).transpose();

		NNModel model = setUp();

		double loss = model.learning(0.00001d, batch(x), batch(y), 100, 200);
		// double loss = model.learning(0.01d, Arrays.asList(x), Arrays.asList(y), 1,
		// 1000);
		ts = System.currentTimeMillis() - ts;
		String m = "Total Loss: " + loss + "\n";
		System.out.println(m + "Total time elapsed: " + ts + "ms");

		List<RealMatrix> ret = model.predict(Arrays.asList(x));
		RealMatrix r = ret.get(0);
		for (int i = 0; i < ret.get(0).getColumnDimension(); i++) {
			double[] X = x.getColumn(i);
			double[] Y = y.getColumn(i);
			double p = r.getColumnVector(i).getEntry(0);
			System.out.println(Arrays.toString(X) + "=>" + Arrays.toString(Y) + "  ----->  " + p);
		}
	}

	static List<RealMatrix> batch(RealMatrix x) {
		List<RealMatrix> ret = new ArrayList<>();
		for (int i = 0; i < x.getColumnDimension(); i++) {
			RealMatrix r = MatrixUtils.createRealMatrix(x.getRowDimension(), 1);
			r.setColumnVector(0, x.getColumnVector(i));
			ret.add(r);
		}
		return ret;
	}

	static NNModel setUp() {
		// Xavier.debug = 0.1d;
		NNModel.resetLogging();
		// NNModel.setLoggingDebugMode(true);
		// TracedComputation.debugLevel(2);
		// RMSPropOptimizer.debugLevel(2);

		DenseLayer l0 = new DenseLayer(2, 3, "Layer0");
		DenseLayer l1 = new DenseLayer(3, 1, "Layer1");

		// l0.setBiased(0.1d);
		// l1.setBiased(0.1d);

		l0.setActivationName("sigmoid");
		l1.setActivationName("sigmoid");

		NNModel model = new NNModel(0.3d, 0.95d);
		model.setMinimumError(0.001d);
		model.addLayer(l0);
		model.addLayer(l1);

		model.setLossName("mse");
		System.out.println(model.debugInfo());
		return model;
	}

}
