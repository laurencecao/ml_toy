package dl.nn;

import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.linear.DefaultRealMatrixChangingVisitor;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class NN {

	static boolean in_debug = false;

	int[] neuralByLayer;
	RealMatrix[] weights;

	public NN(int[] nodes) {
		neuralByLayer = nodes;
		init();
	}

	void init() {
		weights = new RealMatrix[neuralByLayer.length - 1];
		Random rnd = new Random(5321);
		for (int i = 0; i < neuralByLayer.length - 1; i++) {
			int from = neuralByLayer[i];
			int to = neuralByLayer[i + 1];

			weights[i] = MatrixUtils.createRealMatrix(to, from + 1);
			weights[i].walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
				@Override
				public double visit(int row, int column, double value) {
					return rnd.nextDouble();
				}
			});
		}
	}

	RealMatrix getWeight(int from) {
		return weights[from];
	}

	void setWeight(int from, RealMatrix m) {
		weights[from] = m;
	}

	void debugInfo(RealVector[] a, RealVector[] z, RealVector[] err) {
		if (!in_debug) {
			return;
		}
		System.out.println("..................................");
		System.out.println("Total Neural Nodes: " + Arrays.toString(neuralByLayer));
		int sz = weights.length;
		if (a != null) {
			sz = sz < a.length ? a.length : sz;
		}
		for (int i = 0; i < sz; i++) {
			if (i < weights.length) {
				System.out.println("Layer[" + i + "] weights -----> " + weights[i]);
			}
			if (a != null) {
				System.out.println("Layer[" + i + "] summary -----> " + a[i]);
			}
			if (z != null) {
				System.out.println("Layer[" + i + "] activation -----> " + z[i]);
			}
			if (err != null) {
				System.out.println("Layer[" + i + "] error -----> " + err[i]);
			}
		}
		System.out.println("..................................");
	}

}
