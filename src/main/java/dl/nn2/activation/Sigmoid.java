package dl.nn2.activation;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import org.apache.commons.math3.util.FastMath;

public class Sigmoid implements GateFunction {

	@Override
	public RealMatrix forward(RealMatrix m) {
		return eval(m, true);
	}

	@Override
	public RealMatrix backward(RealMatrix m) {
		return eval(m, false);
	}

	protected RealMatrix eval(RealMatrix m, boolean forward) {
		RealMatrix ret = m.copy();
		ret.walkInOptimizedOrder(new RealMatrixChangingVisitor() {

			@Override
			public double visit(int row, int column, double value) {
				double a = FastMath.exp(-1d * value);
				double b = 1d / (1d + a);
				if (forward) {
					return b;
				} else {
					return b * (1 - b);
				}
			}

			@Override
			public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
			}

			@Override
			public double end() {
				return 0;
			}
		});
		return ret;
	}

	@Override
	public String name() {
		return Sigmoid.class.getSimpleName();
	}

}
