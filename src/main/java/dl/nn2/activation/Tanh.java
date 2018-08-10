package dl.nn2.activation;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import org.apache.commons.math3.util.FastMath;

public class Tanh implements GateFunction {

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
				double a = FastMath.exp(value);
				double b = FastMath.exp(-1d * value);
				double tanh = (a - b) / (a + b);
				if (forward) {
					return tanh;
				} else {
					return 1 - tanh * tanh;
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
		return Tanh.class.getSimpleName();
	}

}
