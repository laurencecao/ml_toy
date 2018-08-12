package dl.nn2.loss;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealVectorChangingVisitor;
import org.apache.commons.math3.util.FastMath;

import dl.nn2.activation.Softmax;
import dl.nn2.graph.MatrixDataEdge;

public class CategoricalCrossEntropy implements LossFunction {

	@Override
	public double eval(RealMatrix target, RealMatrix y) {
		double ret = 0d;
		for (int i = 0; i < target.getColumnDimension(); i++) {
			RealVector vT = target.getColumnVector(i);
			RealVector vY = y.getColumnVector(i).copy();
			vY.walkInOptimizedOrder(log);
			ret += vT.dotProduct(vY);
		}
		return ret / y.getColumnDimension();
	}

	static RealVectorChangingVisitor log = new RealVectorChangingVisitor() {

		@Override
		public void start(int dimension, int start, int end) {
		}

		@Override
		public double visit(int index, double value) {
			return FastMath.log(value) * -1d;
		}

		@Override
		public double end() {
			return 0;
		}

	};

	@Override
	public String name() {
		return CategoricalCrossEntropy.class.getSimpleName();
	}

	@Override
	public RealMatrix dif(RealMatrix target, RealMatrix y) {
		RealMatrix ret = MatrixUtils.createRealMatrix(y.getRowDimension(), y.getColumnDimension());
		for (int i = 0; i < y.getColumnDimension(); i++) {
			RealVector vT = target.getColumnVector(i);
			RealVector vY = y.getColumnVector(i);
			ret.setColumnVector(i, vT.ebeDivide(vY).mapMultiply(-1d));
		}
		return ret;
	}

	public static void main(String[] args) {
		/**
		 * ∂L/∂z = −∑[(y * 1/p) * ∂p/∂z]
		 * 
		 * ∂L/∂p = −∑[(y * 1/p)]
		 */
		CategoricalCrossEntropy ce = new CategoricalCrossEntropy();
		RealMatrix y = MatrixUtils.createRealMatrix(new double[][] { { 0.29450637, 0.34216758, 0.36332605 } })
				.transpose();
		RealMatrix t = MatrixUtils.createRealMatrix(new double[][] { { 1, 0, 0 } }).transpose();
		double error = ce.eval(t, y);
		System.out.println("Cross Entropy: " + error);
		System.out.println("Derivated Cross Entropy: \n");
		RealMatrix m = ce.dif(t, y);
		System.out.println(MatrixDataEdge.pretty0(m));

		/**
		 * ∂p/∂z = Jacobian(D(y)S(x)) = S(x) * (1 - S(y))
		 */
		Softmax s = new Softmax();
		RealMatrix m1 = s.backward(y);
		System.out.println("Derivated Softmax: \n");
		System.out.println(MatrixDataEdge.pretty0(m1));
		RealMatrix m2 = m1.multiply(m);
		System.out.println("Derivated (Cross Entropy + Softmax): \n");
		System.out.println(MatrixDataEdge.pretty0(m2));

		/**
		 * ∂L/∂z=p−y
		 */
		RealMatrix m3 = y.subtract(t);
		System.out.println("Derivated (Cross Entropy + Softmax): \n");
		System.out.println(MatrixDataEdge.pretty0(m3));
	}

}
