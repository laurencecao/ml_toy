package dl.nn;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealVectorChangingVisitor;
import org.apache.commons.math3.util.FastMath;

import dl.dataset.NNDataset;

public class NNPlayer {

	final static boolean in_debug = true;
	static double alpha = 0.1d;
	static double epi = 0.0001d;

	public static void main(String[] args) {
		// XOR();
		Circle();
	}

	static void XOR() {
		alpha = 0.8d;
		epi = 0.0001d;
		NN inst = build(2, 3, 1);
		inst.debugInfo(null, null, null);
		RealVector[] data = NNDataset.getData(NNDataset.XOR);
		RealVector[] target = NNDataset.getLabel(NNDataset.XOR);
		training(inst, data, target);

		for (int i = 0; i < data.length; i++) {
			RealVector r = predict(inst, data[i], false);
			System.out.println(data[i] + " --> " + r);
		}
	}

	static void Circle() {
		alpha = 0.05d;
		epi = 0.01d;
		NN inst = build(10, 4, 4, 1);
		RealVector[] data = NNDataset.getData(NNDataset.CIRCLE);
		RealVector[] target = NNDataset.getLabel(NNDataset.CIRCLE);
		training(inst, data, target);

		for (int i = 0; i < data.length; i++) {
			RealVector r = predict(inst, data[i], false);
			System.out.println(data[i] + " --> " + r.mapMultiply(4).mapAdd(4));
		}

		RealVector d = MatrixUtils.createRealVector(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 4, 0 });
		RealVector r = predict(inst, d, false);
		System.out.println(d + " --> " + r.mapMultiply(4).mapAdd(4));
	}

	static void training(NN inst, RealVector[] data, RealVector[] target) {
		int sz = inst.neuralByLayer.length;
		double err = 10d;
		List<RealVector[][]> trData = normData(sz, data);
		int loop = 0;
		while (err > epi) {
			RealVector[][] d;
			for (int i = 0; i < data.length; i++) {
				d = trData.get(i);
				forwardProp(inst, d[0], d[1]);
				backProp(inst, d[0], d[1], target[i]);
			}
			for (int i = 0; i < data.length; i++) {
				d = trData.get(i);
				err += MSE(inst, d[0][0], target[i]);
			}
			err /= data.length;
			if (loop % 10000 == 0) {
				System.out.println("MSE ---> " + err);
			}
			loop++;
			// if (loop > 100000) {
			// break;
			// }
		}
		boolean o = NN.in_debug;
		NN.in_debug = true;
		inst.debugInfo(null, null, null);
		NN.in_debug = o;
	}

	static NN build(int... x) {
		return new NN(x);
	}

	static RealVector[][] normData(int sz, RealVector data) {
		RealVector[] a = new RealVector[sz];
		a[0] = MatrixUtils.createRealVector(new double[] { 1 });
		a[0] = a[0].append(data);
		RealVector[] z = new RealVector[sz];
		z[0] = a[0].copy();
		return new RealVector[][] { a, z };
	}

	static List<RealVector[][]> normData(int sz, RealVector[] data) {
		List<RealVector[][]> ret = new ArrayList<RealVector[][]>();
		for (int i = 0; i < data.length; i++) {
			ret.add(normData(sz, data[i]));
		}
		return ret;
	}

	final static RealVectorChangingVisitor activation = new RealVectorChangingVisitor() {
		@Override
		public void start(int dimension, int start, int end) {
		}

		@Override
		public double visit(int index, double value) {
			return FastMath.tanh(value);
		}

		@Override
		public double end() {
			return 0;
		}
	};

	final static RealVectorChangingVisitor deriact = new RealVectorChangingVisitor() {
		@Override
		public void start(int dimension, int start, int end) {
		}

		@Override
		public double visit(int index, double value) {
			return 1 - FastMath.pow(FastMath.tanh(value), 2);
		}

		@Override
		public double end() {
			return 0;
		}
	};

	static void forwardProp(NN inst, RealVector[] a, RealVector[] z) {
		for (int i = 0; i < inst.weights.length; i++) {
			RealMatrix wt = inst.getWeight(i);
			RealVector X = wt.operate(z[i]);
			if (i < inst.weights.length - 1) {
				X = MatrixUtils.createRealVector(new double[] { 1 }).append(X);
			}
			a[i + 1] = X.copy();
			X.walkInOptimizedOrder(activation);
			z[i + 1] = X;
		}
	}

	static void backProp(NN inst, RealVector[] a, RealVector[] z, RealVector target) {
		// hypothesis layer is last layer, but weight's layer number should - 1
		int layer = inst.neuralByLayer.length - 2;
		RealVector[] err = new RealVector[inst.neuralByLayer.length];
		{
			// w = w - alpha * (h - target) * x
			RealMatrix old_w = inst.getWeight(layer);
			RealVector h = z[z.length - 1];
			RealVector x = z[z.length - 2];
			err[layer + 1] = h.subtract(target); // keep error
			RealMatrix _a = MatrixUtils.createColumnRealMatrix(err[layer + 1].toArray());
			RealMatrix _b = MatrixUtils.createRowRealMatrix(x.toArray());
			RealMatrix deri = _a.multiply(_b);
			inst.setWeight(layer, old_w.subtract(deri.scalarMultiply(alpha)));
		}

		inst.debugInfo(a, z, err);
		for (int i = layer - 1; i >= 0; i--) {
			/**
			 * layer derivation = δ * Z
			 * 
			 * w.r.t layer number: j < k
			 * 
			 * δ(j) = ∅'(a(j)) * Σ [ w(k,j)* δ(k)]
			 **/
			RealMatrix w = inst.getWeight(i + 1);
			int c = w.getColumnDimension();
			int r = w.getRowDimension();
			w = w.getSubMatrix(0, r - 1, 1, c - 1);
			RealVector _t1 = a[i + 1].copy().getSubVector(1, c - 1);
			_t1.walkInOptimizedOrder(deriact);
			RealVector _t2 = w.transpose().operate(err[i + 2]);
			err[i + 1] = _t1.ebeMultiply(_t2);

			RealMatrix dw = err[i + 1].outerProduct(z[i]);
			RealMatrix old_w = inst.getWeight(i);
			inst.debugInfo(a, z, err);
			inst.setWeight(i, old_w.subtract(dw.scalarMultiply(alpha)));
		}

	}

	static RealVector predict(NN inst, RealVector data, boolean normed) {
		RealVector d = data;
		if (!normed) {
			d = normData(inst.neuralByLayer.length, d)[0][0];
		}
		for (int i = 0; i < inst.weights.length; i++) {
			RealMatrix w = inst.getWeight(i);
			d = w.operate(d);
			d.walkInOptimizedOrder(activation);
			if (i < inst.weights.length - 1) {
				d = MatrixUtils.createRealVector(new double[] { 1 }).append(d);
			}
		}
		return d;
	}

	static double MSE(NN inst, RealVector data, RealVector target) {
		RealVector v = predict(inst, data, true);
		return v.subtract(target).getNorm();
	}

}
