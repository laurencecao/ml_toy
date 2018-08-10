package dl.nn2.loss;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealVectorChangingVisitor;
import org.apache.commons.math3.util.FastMath;

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
		return ret;
	}

	static RealVectorChangingVisitor log = new RealVectorChangingVisitor() {

		@Override
		public void start(int dimension, int start, int end) {
		}

		@Override
		public double visit(int index, double value) {
			return FastMath.log(value);
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
			RealVector a = vT.ebeDivide(vY).mapMultiply(-1);
			RealVector b = vT.mapAdd(-1).ebeDivide(vY.mapAdd(-1));
			ret.setColumnVector(i, a.add(b));
		}
		return ret;
	}

}
